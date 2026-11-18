"""Select a decoder scale on out-of-fold training emissions, then score student tests."""
import argparse
import json
from pathlib import Path
from time import perf_counter
from zipfile import ZipFile

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from tune import ROOT,columns,load_trips,predict,save_rows
from run_stage3 import metrics,read_array
from decoder import road_transitions,decode

SCALES=(8,20,45)


def point_scores(t,pred,prob=None):
    ids=t["ids"]
    assert np.all((pred[:,None]==ids).any(axis=1)) and (pred>=0).all()
    r=metrics(t["track"],t["arcs"],t["route"],t["label"],t["eligible"],pred[:,None],t["fallback"])
    out={k:r[k] for k in ("gps","eligible","correct","macro_f1","continuous","steps","changes",
                          "connected_changes","sequence_f1","reverse_errors")}
    if prob is not None:
        keep=t["eligible"]
        p=(prob[keep]*(ids[keep]==t["label"][keep,None])).sum(axis=1)
        out["nll"]=float(-np.log(np.maximum(1e-12,p)).sum())
    else:
        out["nll"]=None
    return out


def aggregate(rows):
    def total(k):
        return sum(r[k] for r in rows)
    return dict(gps=total("gps"),eligible=total("eligible"),correct=total("correct"),
        accuracy=total("correct")/total("eligible"),
        macro_f1=np.mean([r["macro_f1"] for r in rows if r["eligible"]]),
        continuity=total("continuous")/total("steps"),
        change_continuity=total("connected_changes")/total("changes"),
        sequence_f1=np.mean([r["sequence_f1"] for r in rows]),
        log_loss=None if rows[0]["nll"] is None else total("nll")/total("eligible"),
        reverse_errors=total("reverse_errors"),seconds=None if rows[0]["seconds"] is None else total("seconds"),
        repaired=total("repaired"),damaged=total("damaged"),
        connectors=total("connectors"),gap_restarts=total("gap_restarts"),
        topology_restarts=total("topology_restarts"))


def measure(split,trip,method,scale,t,pred,seconds,prob=None,base=None,starts=None,bridges=None):
    r=point_scores(t,pred,prob)
    keep=t["eligible"]
    good=pred==t["label"]
    old=good if base is None else base==t["label"]
    gap=np.diff(t["track"][:,2])>30
    return dict(split=split,trip=trip,method=method,scale=scale,**r,seconds=seconds,
                repaired=int((keep & good & ~old).sum()),damaged=int((keep & ~good & old).sum()),
                connectors=0 if bridges is None else int((bridges>=0).sum()),
                gap_restarts=0 if starts is None else int((gap & starts[1:]).sum()),
                topology_restarts=0 if starts is None else int((~gap & starts[1:]).sum()))


def prepare(z,split,k,t):
    nodes=read_array(z,f"Student folder/{split}/{k}/{k}.nodes",float,2)
    return road_transitions(t["track"],nodes,t["arcs"],t["ids"],t["features"])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive",type=Path)
    parser.add_argument("cache",type=Path)
    parser.add_argument("scratch",type=Path)
    args=parser.parse_args()
    rows,saved=[],{}
    with ZipFile(args.archive) as z:
        train=load_trips(args.cache,z,"train")
        with np.load(args.scratch/"best-oof.npz") as oof:
            for k,t in train.items():
                prob=oof[k]
                base=t["ids"][np.arange(len(prob)),prob.argmax(axis=1)]
                rows.append(measure("cv",k,"tuned",0,t,base,None,prob))
                prepared=prepare(z,"train",k,t)
                for scale in SCALES:
                    start=perf_counter()
                    pred,starts,bridges=decode(prepared,t["ids"],prob,scale)
                    rows.append(measure("cv",k,"viterbi",scale,t,pred,perf_counter()-start,
                                        base=base,starts=starts,bridges=bridges))
        configs={s:aggregate([r for r in rows if r["scale"]==s]) for s in SCALES}
        best=max(SCALES,key=lambda s:(configs[s]["sequence_f1"],configs[s]["macro_f1"]))
        config=dict(scale_m=best,selection="maximum mean trip sequence F1 on training out-of-fold emissions",
            scales_tested=list(SCALES),gap_seconds=30,change_penalty=.25,bridge_extra_penalty=.75,
            u_turn_extra_penalty=1.5,exit_log_weight=-3,backward_tolerance_m=5)
        (ROOT/"decoder-config.json").write_text(json.dumps(config,indent=2))
        save_rows(ROOT/"decoder-search.csv",[dict(scale_m=s,**configs[s]) for s in SCALES])
        print("Frozen decoder scale",best,flush=True)
        del train
        test=load_trips(args.cache,z,"student_test")
        for method in ("untuned","tuned"):
            model=joblib.load(ROOT/(method+".joblib"))
            for k,t in test.items():
                start=perf_counter()
                ranked,prob=predict(model,columns("hist_boost"),t["features"],t["ids"])
                elapsed=perf_counter()-start
                base=ranked[:,0]
                rows.append(measure("student_test",k,method,0,t,base,elapsed,prob))
                saved[f"{method}_{k}"]=base
                if method=="tuned":
                    # Total for decoded route includes scoring, geometry and Viterbi, without file IO.
                    nodes=read_array(z,f"Student folder/student_test/{k}/{k}.nodes",float,2)
                    start=perf_counter()
                    prepared=road_transitions(t["track"],nodes,t["arcs"],t["ids"],t["features"])
                    pred,starts,bridges=decode(prepared,t["ids"],prob,best)
                    rows.append(measure("student_test",k,"viterbi",best,t,pred,elapsed+perf_counter()-start,
                                          base=base,starts=starts,bridges=bridges))
                    saved.update({f"viterbi_{k}":pred,f"start_{k}":starts,f"connector_{k}":bridges})
    save_rows(ROOT/"trip-results.csv",rows)
    np.savez_compressed(ROOT/"student-predictions.npz",**saved)
    table=[]
    for split,method,scale in [("cv","tuned",0),("cv","viterbi",best),("student_test","untuned",0),
             ("student_test","tuned",0),("student_test","viterbi",best)]:
        selected=[r for r in rows if (r["split"],r["method"],r["scale"])==(split,method,scale)]
        summary=aggregate(selected)
        table.append(dict(split=split,method=method,**summary))
    save_rows(ROOT/"comparison.csv",table)
    print(table[-1],flush=True)


if __name__=="__main__":
    with threadpool_limits(3):
        main()
