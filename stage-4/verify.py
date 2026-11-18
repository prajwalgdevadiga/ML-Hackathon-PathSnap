"""Cross-check saved fits, out-of-fold assignments and results against cached inputs."""
import argparse
import csv
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import f1_score
from threadpoolctl import threadpool_limits

from compare import ALGORITHMS, FLOOR, columns, predict


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("cache",type=Path)
    p.add_argument("--out",type=Path,default=Path(__file__).parent)
    args=p.parse_args()
    out=args.out
    with (out/"trip-scores.csv").open() as stream:
        records=list(csv.DictReader(stream))
    with (out/"comparison.csv").open() as stream:
        summary={(r["algorithm"],r["split"]):r for r in csv.DictReader(stream)}
    folds=json.loads((out/"folds.json").read_text())
    trip_count=0
    for name in ALGORITHMS:
        model=joblib.load(out/"models"/(name+".joblib"))
        with np.load(out/"predictions"/(name+".npz")) as saved:
            for split,folder in [("cv","train"),("student_test","student_test")]:
                scores=[]
                matches=observations=0
                nll=0.
                rows=[r for r in records if r["algorithm"]==name and r["split"]==split]
                assert len(rows)==len({r["trip"] for r in rows})==(80 if split=="cv" else 10)
                for r in rows:
                    trip=r["trip"]
                    key=f"{split}_{trip}"
                    pred=saved[key]
                    prob=saved[key+"_eligible_reference_p"]
                    with np.load(args.cache/"scoring_only"/folder/(trip+".npz")) as z:
                        keep=z["eligible"]
                        y=z["label"][keep]
                    with np.load(args.cache/folder/(trip+".npz")) as z:
                        ids=z["arc_ids"]
                        assert (ids==pred[:,None]).any(axis=1).all()
                        if split=="student_test":
                            ranked,distribution=predict(model,columns(name),z["features"],ids)
                            assert np.array_equal(ranked[:,0],pred)
                            assert np.allclose((distribution[keep]*(ids[keep]==y[:,None])).sum(axis=1),prob)
                        else:
                            assert trip in folds[r["fold"]]
                    assert len(prob)==len(y) and len(pred)==int(r["gps"])
                    assert np.all((prob>=0)&(prob<=1))
                    score=f1_score(y,pred[keep],average="macro",zero_division=0)
                    assert np.isclose(score,float(r["macro_f1"]))
                    scores.append(score)
                    matches+=int(np.sum(pred[keep]==y))
                    observations+=len(y)
                    nll-=np.log(np.maximum(FLOOR,prob)).sum()
                    trip_count+=1
                report=summary[name,split]
                assert matches==int(report["correct"]) and observations==int(report["eligible"])
                assert np.isclose(np.mean(scores),float(report["macro_f1"]))
                assert np.isclose(nll/observations,float(report["log_loss"]))
    print(f"Verified {trip_count} out-of-fold or student-test trip results and replayed all student-test predictions from the saved models.")


if __name__=="__main__":
    with threadpool_limits(3):
        main()
