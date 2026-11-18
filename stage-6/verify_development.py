"""Recheck the running student rehearsal against the frozen Stage 5 results."""
import argparse
import csv
import json
from pathlib import Path
from urllib.request import ProxyHandler,Request,build_opener

import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("url")
    p.add_argument("stage5",type=Path)
    p.add_argument("--out",type=Path,default=Path(__file__).parent/"evidence"/"api-checks.csv")
    args=p.parse_args()
    http=build_opener(ProxyHandler({}))
    base=args.url.rstrip("/")
    with http.open(base+"/api/trips",timeout=10) as r:
        catalog=json.load(r)
    assert catalog["mode"]=="student-test" and len(catalog["trips"])==10
    with (args.stage5/"trip-results.csv").open() as f:
        scores={r["trip"]:r for r in csv.DictReader(f) if r["split"]=="student_test" and r["method"]=="viterbi"}
    assert set(catalog["trips"])==set(scores)
    rows=[]
    with np.load(args.stage5/"student-predictions.npz") as saved:
        for trip in catalog["trips"]:
            with http.open(Request(base+"/api/trips/"+trip+"/predict",method="POST"),timeout=120) as r:
                payload=json.load(r)
            assert set(payload)=={"request_id","trip","mode","map","metrics","inference_ms"}
            assert set(payload["map"])=={"gps","roads","runs","connectors","predicted_arc_ids"}
            assert set(payload["metrics"])=={"observations","eligible","correct","accuracy","macro_f1","sequence_f1","continuity"}
            pred=payload["map"]["predicted_arc_ids"]
            assert np.array_equal(pred,saved["viterbi_"+trip])
            m=payload["metrics"]
            s=scores[trip]
            assert m["correct"]==int(s["correct"]) and m["eligible"]==int(s["eligible"])
            assert len(pred)==len(payload["map"]["gps"])==m["observations"]
            assert np.isclose(m["accuracy"],int(s["correct"])/int(s["eligible"]))
            assert np.isclose(m["continuity"],int(s["continuous"])/int(s["steps"]))
            for k in ["macro_f1","sequence_f1"]:
                assert np.isclose(m[k],float(s[k]))
            assert len(payload["map"]["runs"])==int(saved["start_"+trip].sum())
            rows.append(dict(trip=trip,request_id=payload["request_id"],http_status=200,**m))
    with args.out.open("w",newline="",encoding="utf8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("All ten live API results matched Stage 5, including every predicted arc and displayed metric.")


if __name__=="__main__":
    main()
