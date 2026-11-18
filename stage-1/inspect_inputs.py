from collections import Counter, defaultdict
from hashlib import sha256
from pathlib import Path, PurePosixPath
from statistics import median
from zipfile import ZipFile

import numpy as np


source = Path(__file__).resolve().parents[1] / "inputs" / "Student folder.zip"
files = defaultdict(dict)
rows = []
with ZipFile(source) as archive:
    assert archive.testzip() is None
    for entry in archive.infolist():
        if entry.is_dir():
            continue
        name = PurePosixPath(entry.filename)
        root, split, trip, filename = name.parts
        assert name.stem == trip and name.suffix not in files[split, trip]
        files[split, trip][name.suffix] = entry
    for (split, trip), members in sorted(files.items()):
        assert set(members) == {".nodes", ".arcs", ".track", ".route"}
        data = {}
        for suffix, cols in [(".nodes", 2), (".arcs", 2), (".track", 3), (".route", 1)]:
            with archive.open(members[suffix]) as stream:
                values = np.loadtxt(stream, ndmin=2)
            assert len(values) and values.shape[1] == cols and np.isfinite(values).all()
            if suffix in [".arcs", ".route"]:
                assert (values == np.trunc(values)).all()
                values = values.astype(np.int64)
            else:
                assert (abs(values[:, 0]) <= 180).all() and (abs(values[:, 1]) <= 90).all()
            data[suffix] = values
        nodes, arcs, track, route = (data[s] for s in (".nodes", ".arcs", ".track", ".route"))
        assert arcs.min() >= 0 and arcs.max() < len(nodes)
        assert route.min() >= 0 and route.max() < len(arcs)
        gaps = np.diff(track[:, 2])
        rows.append(dict(split=split, trip=trip, gps=len(track), nodes=len(nodes), arcs=len(arcs),
                         route=len(route), seconds=int(np.sum(gaps == 1)), pairs=len(gaps),
                         nonpos=int(np.sum(gaps <= 0)), maxgap=float(max(gaps))))

train = {trip for split, trip in files if split == "train"}
student = {trip for split, trip in files if split == "student_test"}
assert train.isdisjoint(student)
report = [
    "# PathSnap | Stage 1 input audit", "",
    "Scope covers the supplied Student folder.zip. Each data file was read. Checks covered archive integrity, trip filenames, expected column counts, finite values, coordinate bounds, integer identifiers and trip-local reference bounds.", "",
    f"Archive SHA-256: `{sha256(source.read_bytes()).hexdigest()}`", "",
    "| Partition | Trips | GPS observations | GPS per trip (min / median / max) | Route entries | 1-second intervals | Non-increasing intervals | Longest interval |",
    "|---|---:|---:|---|---:|---:|---:|---:|",
]
for split in ("train", "student_test"):
    subset = [r for r in rows if r["split"] == split]
    counts = [r["gps"] for r in subset]
    one = sum(r["seconds"] for r in subset)
    total = sum(r["pairs"] for r in subset)
    report.append(f"| {split} | {len(subset)} | {sum(counts):,} | {min(counts):,} / {median(counts):g} / {max(counts):,} | {sum(r['route'] for r in subset):,} | {one:,}/{total:,} ({one/total:.2%}) | {sum(r['nonpos'] for r in subset)} | {max(r['maxgap'] for r in subset):g} s |")
different = sum(r["gps"] != r["route"] for r in rows)
report += ["", f"All {len(files) * 4} data files passed these structural checks. The two partitions have distinct trip IDs. GPS observation and reference route row counts differ in {different} of {len(rows)} trips. Each route lists traversed arcs in order. It has no per-observation timestamps. Observation-level evaluation needs a documented alignment method and an uncertainty check.", "",
           "## Per-trip counts", "", "| Partition | Trip ID | GPS observations | Nodes | Directed arcs | Route entries | Longest GPS interval (s) |", "|---|---|---:|---:|---:|---:|---:|"]
for r in sorted(rows, key=lambda item: (item["split"] != "train", item["trip"])):
    report.append("| {split} | `{trip}` | {gps:,} | {nodes:,} | {arcs:,} | {route:,} | {maxgap:g} |".format(**r))
target = Path(__file__).with_name("Input audit.md")
target.write_text("\n".join(report) + "\n", encoding="utf-8")
print("\n".join(report[:13]))
