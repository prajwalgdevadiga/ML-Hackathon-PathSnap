"""Replay raw unlabeled trips and check graph connections, scoring, gaps and Viterbi."""
import argparse
import csv
import itertools
import json
from pathlib import Path
from zipfile import ZipFile

import joblib
import numpy as np
from sklearn.metrics import f1_score
from threadpoolctl import threadpool_limits

from tune import ROOT
from run_stage3 import read_array
from decoder import viterbi
from inference import predict_trip


def small_cases():
    rng=np.random.default_rng(352)
    ids=np.tile(np.arange(3),(5,1))
    for _ in range(30):
        em=rng.dirichlet([1,1,1],size=5)
        log_mat=np.log(rng.dirichlet([1,1,1],size=(4,3)))
        result,_,starts=viterbi(ids,em,log_mat,np.ones(4))
        def score(states):
            return np.log(em[np.arange(5),states]).sum()+sum(log_mat[j,states[j],states[j+1]] for j in range(4))
        oracle=max(itertools.product(range(3),repeat=5),key=score)
        assert np.array_equal(result,oracle) and starts.sum()==1
    probs=np.ones((5,3))/3
    local=ids.copy()
    local[2]=-1
    transition=np.zeros((4,3,3))
    transition[3]=-np.inf
    pred,_,starts=viterbi(local,probs,transition,np.array([31,1,1,1]))
    assert pred.tolist()==[0,0,-1,0,0] and starts.tolist()==[True,True,False,True,True]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive")
    parser.add_argument("cache",type=Path)
    args=parser.parse_args()
    small_cases()
    tuned=joblib.load(ROOT/"tuned.joblib")
    scale=json.loads((ROOT/"decoder-config.json").read_text())["scale_m"]
    with (ROOT/"trip-results.csv").open() as f:
        rows=[r for r in csv.DictReader(f) if r["split"]=="student_test" and r["method"]=="viterbi"]
    replay_seconds=0
    with ZipFile(args.archive) as z,np.load(ROOT/"student-predictions.npz") as saved:
        for row in rows:
            k=row["trip"]
            base=f"Student folder/student_test/{k}/{k}."
            track,nodes,arcs=[read_array(z,base+s,d,2) for s,d in [("track",float),("nodes",float),("arcs",int)]]
            result=predict_trip(tuned,scale,track,nodes,arcs)
            replay_seconds+=result["seconds"]
            pred,starts,bridge=result["decoded"],result["starts"],result["connectors"]
            for field,key in [("classifier","tuned"),("decoded","viterbi"),("starts","start"),("connectors","connector")]:
                assert np.array_equal(result[field],saved[f"{key}_{k}"])
            assert starts[0] and starts[1:][np.diff(track[:,2])>30].all()
            for i in np.flatnonzero(~starts):
                p,c=pred[i-1],pred[i]
                b=bridge[i]
                if b>=0:
                    assert arcs[p,1]==arcs[b,0] and arcs[b,1]==arcs[c,0]
                else:
                    assert p==c or arcs[p,1]==arcs[c,0]
            with np.load(args.cache/"scoring_only"/"student_test"/(k+".npz")) as scores:
                valid=scores["eligible"]
                label=scores["label"][valid]
                assert (label==pred[valid]).sum()==int(row["correct"])
                assert np.isclose(f1_score(label,pred[valid],average="macro"),float(row["macro_f1"]))
    print(f"Verified all {len(rows)} raw trip replays, every within-run connection, point scores and 30 exhaustive decoder cases. End-to-end compute: {replay_seconds:.3f} s.")


if __name__=="__main__":
    with threadpool_limits(3):
        main()
