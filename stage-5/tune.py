"""Sequential Gaussian-process Bayesian search on the frozen Stage 4 trip folds."""
import argparse
import itertools
import json
import sys
from pathlib import Path
from time import perf_counter
from zipfile import ZipFile

import joblib
import numpy as np
from scipy.special import ndtr
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "shared"))
from compare import columns, estimator, fit_candidates, load_trips, predict, save_rows

SPACE = dict(max_iter=[55,75,100,135], max_leaf_nodes=[9,15,23,31],
             learning_rate=[.05,.075,.10,.14], l2_regularization=[.1,1,4,12])
INITIAL = [(75,15,.1,1),(55,31,.14,4),(135,9,.075,.1)]
BUDGET = 10


def macro_f1(target, predicted):
    classes = np.union1d(target,predicted)
    a,b = np.searchsorted(classes,target),np.searchsorted(classes,predicted)
    counts = np.bincount(a,minlength=len(classes)) + np.bincount(b,minlength=len(classes))
    correct = np.bincount(a[a==b],minlength=len(classes))
    return float(np.mean(2*correct/counts))


def propose(grid, coords, evaluated, values):
    # Normalised ordinal coordinates for the four discrete search dimensions.
    gp = GaussianProcessRegressor(kernel=Matern(length_scale=.65,nu=2.5),
        optimizer=None,alpha=1e-4,normalize_y=True)
    gp.fit(coords[evaluated],values)
    mean,std = gp.predict(coords,return_std=True)
    improvement = mean-max(values)-.0001
    z = improvement / np.maximum(std,1e-12)
    ei = improvement*ndtr(z) + std*np.exp(-z*z/2)/np.sqrt(2*np.pi)
    ei[evaluated] = -np.inf
    i = int(np.argmax(ei))
    return i,float(ei[i])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive",type=Path)
    parser.add_argument("cache",type=Path)
    parser.add_argument("scratch",type=Path)
    args=parser.parse_args()
    args.scratch.mkdir(parents=True,exist_ok=True)
    with ZipFile(args.archive) as z:
        train=load_trips(args.cache,z,"train")
    folds=json.loads((ROOT/"shared/folds.json").read_text())
    assert sorted(k for f in folds.values() for k in f)==sorted(train) and len(train)==80
    datasets=[]
    for val in folds.values():
        x,y=fit_candidates(train,sorted(set(train)-set(val)))
        datasets.append((x[:,columns("hist_boost")],y,val))
    grid=list(itertools.product(*SPACE.values()))
    coords=np.array(list(itertools.product(range(4),repeat=4)))/3
    evaluated,values,trials=[],[],[]
    best=-np.inf
    started=perf_counter()
    for j in range(BUDGET):
        start=perf_counter()
        if j<len(INITIAL):
            index,ei=grid.index(INITIAL[j]),None
        else:
            index,ei=propose(grid,coords,evaluated,values)
        params=dict(zip(SPACE,grid[index]))
        oof,fold_scores={},[]
        for x,y,val in datasets:
            model=estimator("hist_boost").set_params(**params).fit(x,y)
            scores=[]
            for k in val:
                t=train[k]
                ranked,p=predict(model,columns("hist_boost"),t["features"],t["ids"])
                keep=t["eligible"]
                if keep.any():
                    scores.append(macro_f1(t["label"][keep],ranked[keep,0]))
                oof[k]=p
            fold_scores.append(float(np.mean(scores)))
        score=float(np.mean(fold_scores))
        if score>best:
            best,winning=score,params
            np.savez_compressed(args.scratch/"best-oof.npz",**oof)
        evaluated.append(index)
        values.append(score)
        trials.append(dict(trial=j+1,proposal="initial" if ei is None else "GP expected improvement",
            expected_improvement=ei,**params,**{f"fold_{i+1}":v for i,v in enumerate(fold_scores)},
            cv_macro_f1=score,seconds=perf_counter()-start))
        save_rows(ROOT/"bayesian-trials.csv",trials)
        print(f"Trial {j+1}/{BUDGET}: {score:.5f}, best {best:.5f}",flush=True)
    x,y=fit_candidates(train,sorted(train))
    start=perf_counter()
    model=estimator("hist_boost").set_params(**winning).fit(x[:,columns("hist_boost")],y)
    fit_time=perf_counter()-start
    joblib.dump(model,ROOT/"tuned.joblib",compress=3)
    config=dict(space=SPACE,trials=BUDGET,folds=5,objective="mean trip-local out-of-fold Macro F1",
        selected=winning,selected_cv_macro_f1=best,refit_seconds=fit_time,search_and_refit_seconds=perf_counter()-started)
    (ROOT/"tuning.json").write_text(json.dumps(config,indent=2))


if __name__=="__main__":
    with threadpool_limits(3):
        main()
