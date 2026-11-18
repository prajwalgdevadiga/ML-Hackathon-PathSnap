"""Compare candidate classifiers on common, trip-disjoint folds and development data."""
import argparse
import csv
import json
import platform
import sys
from pathlib import Path
from time import perf_counter
from zipfile import ZipFile

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parent / "stage3"))
from candidates import FEATURES
from run_stage3 import metrics, read_array, save_rows

SEED = 352
FLOOR = 1e-12
ALGORITHMS = ("logistic", "gaussian_nb", "tree", "forest", "hist_boost", "logistic_no_heading")


def estimator(name):
    if name.startswith("logistic"):
        return make_pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=250, random_state=SEED))
    if name == "gaussian_nb":
        return make_pipeline(StandardScaler(), GaussianNB(var_smoothing=1e-8))
    if name == "tree":
        return DecisionTreeClassifier(max_depth=12,min_samples_leaf=30,random_state=SEED)
    if name == "forest":
        return RandomForestClassifier(n_estimators=48,max_depth=14,min_samples_leaf=12,
            max_features="sqrt",max_samples=0.5,n_jobs=3,random_state=SEED)
    if name == "hist_boost":
        return HistGradientBoostingClassifier(max_iter=75,learning_rate=0.1,max_leaf_nodes=15,
            min_samples_leaf=30,l2_regularization=1,early_stopping=False,random_state=SEED)
    raise ValueError(name)


def columns(name):
    if name == "logistic_no_heading":
        return [i for i,k in enumerate(FEATURES) if k not in ("heading_cosine","heading_available")]
    return list(range(len(FEATURES)))


def load_trips(cache, archive, split):
    result = {}
    for f in sorted((cache/split).glob("*.npz")):
        trip=f.stem
        with np.load(f) as z:
            assert tuple(z["feature_names"]) == FEATURES
            ids, features, fallback = z["arc_ids"],z["features"],z["fallback"]
        with np.load(cache/"scoring_only"/split/f.name) as z:
            label, eligible=z["label"],z["eligible"]
        base=f"Student folder/{split}/{trip}/{trip}."
        result[trip]=dict(ids=ids,features=features,fallback=fallback,label=label,eligible=eligible,
            track=read_array(archive,base+"track",float,2),arcs=read_array(archive,base+"arcs",int,2),
            route=read_array(archive,base+"route",int,1))
    return result


def fit_candidates(trips, keys):
    xx,yy=[],[]
    for key in keys:
        t=trips[key]
        valid=(t["ids"]>=0) & t["eligible"][:,None]
        xx.append(t["features"][valid])
        yy.append((t["ids"]==t["label"][:,None])[valid])
    return np.concatenate(xx),np.concatenate(yy)


def predict(model, cols, features, ids):
    """One distribution per observation. The function has no scoring inputs."""
    valid=ids>=0
    positive=np.clip(model.predict_proba(features[valid][:,cols])[:,1].astype(float),FLOOR,1-FLOOR)
    odds=np.zeros(ids.shape)
    odds[valid]=positive/(1-positive)
    probs=odds/odds.sum(axis=1,keepdims=True)
    assert np.isfinite(probs).all() and np.allclose(probs.sum(axis=1),1)
    order=np.argsort(-probs,axis=1,kind="stable")
    return np.take_along_axis(ids,order,axis=1),probs


def score_trips(name, model, trips, keys, split, fold, saved):
    rows=[]
    for key in keys:
        t=trips[key]
        start=perf_counter()
        ranked,probs=predict(model,columns(name),t["features"],t["ids"])
        seconds=perf_counter()-start
        eligible=t["eligible"]
        truth=t["label"][eligible]
        top=ranked[eligible,0]
        reference_prob=np.sum(probs[eligible]*(t["ids"][eligible]==truth[:,None]),axis=1)
        nll=-np.log(np.maximum(FLOOR,reference_prob)).sum()
        r=metrics(t["track"],t["arcs"],t["route"],t["label"],eligible,ranked,t["fallback"])
        classes=np.union1d(top,truth)
        y=np.searchsorted(classes,truth)
        h=np.searchsorted(classes,top)
        a=np.bincount(y,minlength=len(classes))
        b=np.bincount(h,minlength=len(classes))
        hit=np.bincount(y[y==h],minlength=len(classes))
        r.update(macro_precision=float(np.mean(hit/np.maximum(b,1))) if len(truth) else None,
                 macro_recall=float(np.mean(hit/np.maximum(a,1))) if len(truth) else None)
        rows.append(dict(algorithm=name,split=split,fold=fold,trip=key,**r,
                         nll_sum=float(nll),inference_s=seconds))
        if split!="refit_train":
            saved[f"{split}_{key}"]=ranked[:,0]
            saved[f"{split}_{key}_eligible_reference_p"]=reference_prob
    return rows


def aggregate(rows):
    def total(key):
        return sum(r[key] for r in rows)
    n=total("eligible")
    return dict(trips=len(rows),gps=total("gps"),eligible=n,correct=total("correct"),
        accuracy=total("correct")/n,macro_f1=np.mean([r["macro_f1"] for r in rows if r["eligible"]]),
        macro_precision=np.mean([r["macro_precision"] for r in rows if r["eligible"]]),
        macro_recall=np.mean([r["macro_recall"] for r in rows if r["eligible"]]),
        log_loss=total("nll_sum")/n,miss_rate=1-total("hits_at_8")/n,
        continuity=total("continuous")/total("steps"),change_continuity=total("connected_changes")/total("changes"),
        sequence_f1=np.mean([r["sequence_f1"] for r in rows]),reverse_errors=total("reverse_errors"),
        inference_s=total("inference_s"))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive",type=Path)
    parser.add_argument("cache",type=Path)
    parser.add_argument("--out",type=Path,default=Path(__file__).parent)
    args=parser.parse_args()
    out=args.out
    (out/"models").mkdir(parents=True,exist_ok=True)
    (out/"predictions").mkdir(exist_ok=True)
    with ZipFile(args.archive) as z:
        train=load_trips(args.cache,z,"train")
        test=load_trips(args.cache,z,"student_test")
    keys=np.array(sorted(train))
    assert len(keys)==80 and len(test)==10 and set(train).isdisjoint(test)
    folds=[(keys[a].tolist(),keys[b].tolist()) for a,b in KFold(5,shuffle=True,random_state=SEED).split(keys)]
    assert sorted(t for _,val in folds for t in val)==sorted(train)
    for fit,val in folds:
        assert set(fit).isdisjoint(val)
    (out/"folds.json").write_text(json.dumps({str(i):v for i,(_,v) in enumerate(folds,1)},indent=2))
    configs={name:{"parameters":{k:str(v) for k,v in estimator(name).get_params().items() if k!="steps"},
                   "features":[FEATURES[i] for i in columns(name)]} for name in ALGORITHMS}
    (out/"configuration.json").write_text(json.dumps(dict(seed=SEED,sklearn=sklearn.__version__,
        python=platform.python_version(),numpy=np.__version__,algorithms=configs),indent=2))
    comparisons,fold_rows,all_rows,fit_log=[],[],[],[]
    for name in ALGORITHMS:
        rows,saved=[],{}
        for i,(fit,val) in enumerate(folds+[(keys.tolist(),[])],1):
            x,y=fit_candidates(train,fit)
            model=estimator(name)
            x=x[:,columns(name)]
            start=perf_counter()
            model.fit(x,y)
            seconds=perf_counter()-start
            fit_log.append(dict(algorithm=name,fit="full" if i==6 else i,trips=len(fit),candidate_rows=len(y),
                                positives=int(y.sum()),seconds=seconds))
            del x,y
            if i<=5:
                result=score_trips(name,model,train,val,"cv",i,saved)
                fold_rows.append(dict(algorithm=name,fold=i,**aggregate(result)))
                rows.extend(result)
            else:
                joblib.dump(model,out/"models"/(name+".joblib"),compress=3)
                rows.extend(score_trips(name,model,train,keys.tolist(),"refit_train",0,saved))
                rows.extend(score_trips(name,model,test,sorted(test),"student_test",0,saved))
            print(name,"fit",i,"of 6 complete",flush=True)
        for split in ("cv","refit_train","student_test"):
            subset=[r for r in rows if r["split"]==split]
            comparisons.append(dict(algorithm=name,split=split,**aggregate(subset)))
        all_rows.extend(rows)
        np.savez_compressed(out/"predictions"/(name+".npz"),**saved)
        save_rows(out/"comparison.csv",comparisons)
        save_rows(out/"fold-scores.csv",fold_rows)
        save_rows(out/"trip-scores.csv",all_rows)
        save_rows(out/"fit-log.csv",fit_log)
        print(name, comparisons[-1],flush=True)


if __name__=="__main__":
    with threadpool_limits(limits=3):
        main()
