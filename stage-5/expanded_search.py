"""Extend the frozen trip-fold search with regularization and decoder settings."""
import argparse
import json
from pathlib import Path
from time import perf_counter
from zipfile import ZipFile

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from tune import ROOT, columns, estimator, fit_candidates, load_trips, macro_f1, predict, save_rows
from evaluate import aggregate, measure
from decoder import decode, road_transitions
from run_stage3 import read_array


MODEL_DEFAULTS = dict(max_depth=None,min_samples_leaf=30,max_bins=255,max_features=1.)
DECODER_DEFAULTS = dict(scale=8,backward_tolerance=5,change_penalty=.25,bridge_penalty=.75,
                        reverse_penalty=1.5,gap_seconds=30,exit_log_weight=-3.,emission_temperature=1.,
                        max_route_speed=60,gps_distance_slack=80,speed_distance_slack=30,allow_bridge=True)
MODEL_EXTRA = [
    dict(max_depth=5), dict(max_depth=8), dict(min_samples_leaf=60),
    dict(min_samples_leaf=100), dict(max_bins=127), dict(max_bins=63),
    dict(max_features=.75), dict(max_features=.5),
    dict(max_depth=7,min_samples_leaf=60,max_bins=127,max_features=.75),
    dict(max_depth=8,min_samples_leaf=100,max_bins=63,max_features=.85),
]
DECODER_EXTRA = [
    dict(scale=5), dict(scale=12), dict(backward_tolerance=2), dict(backward_tolerance=10),
    dict(change_penalty=.5), dict(bridge_penalty=1.25), dict(reverse_penalty=2.5),
    dict(gap_seconds=20), dict(gap_seconds=45), dict(exit_log_weight=-4.),
    dict(emission_temperature=1.3),dict(emission_temperature=.8),
    dict(max_route_speed=35),dict(gps_distance_slack=45),dict(speed_distance_slack=15),
    dict(allow_bridge=False),
    dict(scale=6,backward_tolerance=3,change_penalty=.4,bridge_penalty=1.,
         reverse_penalty=2.,gap_seconds=25,exit_log_weight=-3.5,emission_temperature=1.15),
]


def model_search(train,folds,out):
    original=json.loads((ROOT/"tuning.json").read_text())["selected"]
    fitted_folds=[]
    for holdout in folds:
        x,y=fit_candidates(train,sorted(set(train)-set(holdout)))
        fitted_folds.append((x,y,holdout))
    names=columns("hist_boost")
    trials=[]
    for round_id, variation in enumerate([{},*MODEL_EXTRA]):
        start=perf_counter()
        settings=original|MODEL_DEFAULTS|variation
        oof={}
        per_trip=[]
        total_loss=0.
        total=0
        correct=0
        fold_scores=[]
        for x,y,holdout in fitted_folds:
            model=estimator("hist_boost").set_params(**settings).fit(x,y)
            scores=[]
            for trip in holdout:
                t=train[trip]
                ids,prob=predict(model,names,t["features"],t["ids"])
                keep=t["eligible"]
                label=t["label"][keep]
                if label.size:
                    scores.append(macro_f1(label,ids[keep,0]))
                    mass=(prob[keep]*(t["ids"][keep]==label[:,None])).sum(axis=1)
                    total_loss-=np.log(np.maximum(mass,1e-12)).sum()
                    total+=len(label)
                    correct+=np.sum(ids[keep,0]==label)
                oof[trip]=prob
            fold_scores.append(float(np.mean(scores)))
            per_trip.extend(scores)
        record=dict(trial=round_id,**settings,
                    **{f"fold_{i+1}":score for i,score in enumerate(fold_scores)},
                    cv_f1=float(np.mean(per_trip)),cv_log_loss=float(total_loss/total),
                    cv_accuracy=float(correct/total),seconds=perf_counter()-start)
        trials.append(record)
        # Prefer reliable emission probabilities when Macro F1 is effectively tied.
        best_f1=max(r["cv_f1"] for r in trials)
        active=min((r for r in trials if r["cv_f1"]>=best_f1-.002),
                   key=lambda r:(r["cv_log_loss"],-r["cv_f1"]))
        np.savez_compressed(out/f"emissions_{round_id:02}.npz",**oof)
        save_rows(out/"classifier-trials.csv",trials)
        print("Classifier",round_id, f"F1={record['cv_f1']:.5f}", f"NLL={record['cv_log_loss']:.5f}",
              "leader",active["trial"],flush=True)
    settings={key:active[key] for key in original|MODEL_DEFAULTS}
    x,y=fit_candidates(train,sorted(train))
    model=estimator("hist_boost").set_params(**settings).fit(x,y)
    joblib.dump(model,out/"candidate.joblib",compress=3)
    return model,active,trials


def variants():
    return [DECODER_DEFAULTS|row for row in [{},*DECODER_EXTRA]]


def node_array(archive,split,trip):
    return read_array(archive,f"Student folder/{split}/{trip}/{trip}.nodes",float,2)


def decode_trip(t,nodes,prob,settings):
    kwargs={k:settings[k] for k in ("scale","gap_seconds","exit_log_weight","emission_temperature")}
    transition={k:v for k,v in settings.items() if k not in kwargs}
    prepared=road_transitions(t["track"],nodes,t["arcs"],t["ids"],t["features"],**transition)
    return decode(prepared,t["ids"],prob,**kwargs)


def decoder_search(train,archive,oof_path,out):
    results=[[] for _ in variants()]
    with np.load(oof_path) as emissions:
        for trip,t in train.items():
            prob=emissions[trip]
            nodes=node_array(archive,"train",trip)
            for i,config in enumerate(variants()):
                start=perf_counter()
                pred,starts,bridges=decode_trip(t,nodes,prob,config)
                results[i].append(measure("cv",trip,"viterbi",config["scale"],t,pred,
                                          perf_counter()-start,starts=starts,bridges=bridges))
    rows=[dict(trial=i,**cfg,**aggregate(result))
          for i,(cfg,result) in enumerate(zip(variants(),results,strict=True))]
    base=rows[0]
    # Maximize sequence score subject to a small non-inferiority margin for points.
    eligible=[r for r in rows if r["accuracy"]>=base["accuracy"]-.001]
    best=max(eligible,key=lambda r:(r["sequence_f1"],r["macro_f1"]))
    save_rows(out/"decoder-trials.csv",rows)
    return best


def student_results(test,archive,models,out):
    rows=[]
    for name,model,settings in models:
        for trip,t in test.items():
            start=perf_counter()
            ranked,prob=predict(model,columns("hist_boost"),t["features"],t["ids"])
            rows.append(measure("student_test",trip,name+" classifier",0,t,ranked[:,0],
                                perf_counter()-start,prob=prob))
            nodes=node_array(archive,"student_test",trip)
            pred,starts,bridges=decode_trip(t,nodes,prob,settings)
            rows.append(measure("student_test",trip,name+" Viterbi",settings["scale"],t,pred,
                                perf_counter()-start,base=ranked[:,0],starts=starts,bridges=bridges))
    save_rows(out/"student-trip-results.csv",rows)
    methods=list(dict.fromkeys(r["method"] for r in rows))
    save_rows(out/"student-comparison.csv",
              [dict(method=m,**aggregate([r for r in rows if r["method"]==m])) for m in methods])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("archive",type=Path)
    p.add_argument("cache",type=Path)
    p.add_argument("out",type=Path)
    p.add_argument("scratch",type=Path)
    args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    args.scratch.mkdir(parents=True,exist_ok=True)
    with ZipFile(args.archive) as archive:
        train=load_trips(args.cache,archive,"train")
        folds=list(json.loads((ROOT/"shared/folds.json").read_text()).values())
        assert sorted(k for val in folds for k in val)==sorted(train) and len(train)==80
        model,leader,trials=model_search(train,folds,args.scratch)
        save_rows(args.out/"classifier-trials.csv",trials)
        choice=decoder_search(train,archive,args.scratch/f"emissions_{leader['trial']:02}.npz",args.out)
        chosen={k:choice[k] for k in DECODER_DEFAULTS}
        (args.out/"selection.json").write_text(json.dumps(dict(classifier=leader,decoder=choice),indent=2))
        joblib.dump(model,args.out/"candidate.joblib",compress=3)
        print("Frozen classifier",leader["trial"],"decoder",choice["trial"],flush=True)
        test=load_trips(args.cache,archive,"student_test")
        student_results(test,archive,[("Current",joblib.load(ROOT/"tuned.joblib"),DECODER_DEFAULTS),
                                      ("Expanded",model,chosen)],args.out)


if __name__=="__main__":
    with threadpool_limits(3):
        main()
