"""Run the fixed nearest-road baseline and audit the approved reference alignment."""
import argparse
import csv
import json
from pathlib import Path, PurePosixPath
from time import perf_counter
from zipfile import ZipFile

import numpy as np
from candidates import CAP, FEATURES, generate
from reference import inspect


def lcs_length(first, second):
    # Each bit represents one reference position; update one full LCS row per token.
    masks = {}
    for j, token in enumerate(second):
        masks[token] = masks.get(token, 0) | (1 << j)
    state = 0
    for token in first:
        choices = masks.get(token, 0) | state
        state = choices & ~(choices - ((state << 1) | 1))
    return state.bit_count()


def metrics(track, arcs, route, labels, eligible, ids, fallback):
    pred = ids[:,0]
    y, h = labels[eligible], pred[eligible]
    classes = np.union1d(y, h)
    idx = np.searchsorted(classes, y)
    actual = np.bincount(idx, minlength=len(classes))
    guessed = np.bincount(np.searchsorted(classes, h), minlength=len(classes))
    tp = np.bincount(idx[y == h], minlength=len(classes))
    matched = ids[eligible] == y[:,None]
    ok_step = np.diff(track[:,2]) <= 30
    change = pred[1:] != pred[:-1]
    direct = arcs[pred[:-1],1] == arcs[pred[1:],0]
    ref_break = arcs[route[:-1],1] != arcs[route[1:],0]
    ref_tokens = route[np.r_[True, (route[1:] != route[:-1]) | ref_break]].tolist()
    pred_tokens = pred[np.r_[True, change | ~ok_step]].tolist()
    lcs = lcs_length(pred_tokens, ref_tokens)
    reverse = (arcs[h] == arcs[y][:,::-1]).all(axis=1) & (h != y)
    return dict(gps=len(track), eligible=len(y), candidates=int((ids >= 0).sum()),
                fallbacks=int(fallback.sum()), correct=int((y==h).sum()), reverse_errors=int(reverse.sum()),
                hits_at_2=int(matched[:,:2].any(axis=1).sum()),
                hits_at_4=int(matched[:,:4].any(axis=1).sum()),hits_at_8=int(matched.any(axis=1).sum()),
                macro_f1=float(np.mean(2*tp/(actual+guessed))) if len(y) else None,
                steps=int(ok_step.sum()), continuous=int((ok_step & (~change | direct)).sum()),
                changes=int((ok_step & change).sum()), connected_changes=int((ok_step & change & direct).sum()),
                predicted_tokens=len(pred_tokens), reference_tokens=len(ref_tokens), lcs=lcs,
                sequence_f1=2*lcs/(len(pred_tokens)+len(ref_tokens)))


def save_rows(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarise(rows):
    result = {key:sum(r[key] for r in rows) for key in ("gps","eligible","candidates","fallbacks",
              "correct","reverse_errors","hits_at_2","hits_at_4","hits_at_8","steps","continuous",
              "changes","connected_changes","inference_s")}
    result["trip_mean_macro_f1"] = float(np.mean([r["macro_f1"] for r in rows if r["macro_f1"] is not None]))
    result["trip_mean_sequence_f1"] = float(np.mean([r["sequence_f1"] for r in rows]))
    return result


def read_array(archive, name, dtype, dims):
    with archive.open(name) as stream:
        return np.loadtxt(stream, dtype=dtype, ndmin=dims)


def brute_check(track, nodes, arcs, ids, features, fallback, trip):
    # Independent all-arc distance calculation on reproducible real observations.
    from geometry import local_metres
    origin = track[:,:2].mean(axis=0)
    xy = local_metres(nodes, origin)
    a, b = xy[arcs.min(axis=1)], xy[arcs.max(axis=1)]
    delta = b-a
    norm = (delta*delta).sum(axis=1)
    rng = np.random.default_rng(int(trip))
    sample = np.unique(np.r_[0,len(track)-1,rng.choice(len(track),12,replace=False),
                             np.flatnonzero(fallback)[:2]])
    for i in sample:
        point = local_metres(track[i,:2],origin)
        t = np.clip(((point-a)*delta).sum(axis=1)/np.where(norm>0,norm,1),0,1)
        dist = np.linalg.norm((point-a) - t[:,None]*delta,axis=1)
        order = np.argsort(dist,kind="stable")
        expected = order[dist[order]<=50][:CAP]
        if not len(expected):
            expected = order[:1]
        actual = ids[i,ids[i]>=0]
        assert np.array_equal(expected, actual), (trip,i,actual,expected)
        assert np.allclose(dist[actual],features[i,:len(actual),0],atol=1e-4)
    return len(sample)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive",type=Path)
    parser.add_argument("--out",type=Path,default=Path(__file__).parent)
    parser.add_argument("--cache",type=Path,help="Optional local cache for later model comparisons")
    args = parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    results, examples, summary = [], [], {}
    predictions = {}
    with ZipFile(args.archive) as archive:
        for split, count in (("train",80),("student_test",10)):
            prefix = f"Student folder/{split}/"
            trips = sorted(PurePosixPath(n).stem for n in archive.namelist() if n.startswith(prefix) and n.endswith(".track"))
            assert len(set(trips)) == len(trips) == count
            for trip in trips:
                base = prefix+trip+"/"+trip+"."
                track = read_array(archive,base+"track",float,2)
                nodes = read_array(archive,base+"nodes",float,2)
                arcs = read_array(archive,base+"arcs",int,2)
                start = perf_counter()
                ids, features, fallback = generate(track,nodes,arcs)
                elapsed = perf_counter()-start
                predictions[f"{split}_{trip}"] = ids[:,0].copy()
                assert (ids[:,0]>=0).all() and np.isfinite(features).all()
                checked = brute_check(track,nodes,arcs,ids,features,fallback,trip)
                # Predictions and all model features already exist before reading labels.
                route = read_array(archive,base+"route",int,1)
                _, (path, _, _, _, eligible, _) = inspect(trip,(track,nodes,arcs,route))
                labels = route[path]
                row = metrics(track,arcs,route,labels,eligible,ids,fallback)
                results.append(dict(split=split,trip=trip,**row,inference_s=elapsed,brute_checks=checked))
                if args.cache:
                    folder = args.cache/split
                    folder.mkdir(parents=True,exist_ok=True)
                    np.savez_compressed(folder/(trip+".npz"),arc_ids=ids,features=features,fallback=fallback,
                                        feature_names=FEATURES)
                    scores = args.cache/"scoring_only"/split
                    scores.mkdir(parents=True,exist_ok=True)
                    np.savez_compressed(scores/(trip+".npz"),label=labels,eligible=eligible)
                miss = eligible & ~(ids == labels[:,None]).any(axis=1)
                wrong = eligible & (ids[:,0] != labels)
                reverse = (arcs[ids[:,0]] == arcs[labels][:,::-1]).all(axis=1)
                for kind,mask in [("reference outside shortlist",miss),
                                  ("opposite direction",wrong & reverse & ~miss),
                                  ("other road ranked first",wrong & ~reverse & ~miss)]:
                    found = np.flatnonzero(mask)
                    if len(found):
                        i = found[0]
                        examples.append(dict(split=split,trip=trip,gps_row=int(i),kind=kind,
                            nearest_arc=int(ids[i,0]),reference_arc=int(labels[i]),
                            nearest_distance_m=float(features[i,0,0]),
                            candidate_ids=" ".join(map(str,ids[i,ids[i]>=0]))))
            split_rows=[r for r in results if r["split"]==split]
            summary[split] = summarise(split_rows)
            print(split, json.dumps(summary[split]), flush=True)
    save_rows(args.out/"per-trip-results.csv", results)
    save_rows(args.out/"error-examples.csv", examples)
    np.savez_compressed(args.out/"baseline-predictions.npz", **predictions)
    (args.out/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf8")


if __name__ == "__main__":
    main()
