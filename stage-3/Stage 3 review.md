# PathSnap candidate generation and nearest-road baseline

The fixed shortlist found the derived reference segment for 13,122 of 13,126 eligible student-test observations. Choosing the nearest segment gave 7,808 correct predictions. Of the 5,318 errors, 4,741 selected a directed arc with the same endpoints as the reference in reverse order. This makes direction of travel a concrete question for the five-model comparison.

## Measured results

| Measure | Training, 80 trips | Student testing, 10 trips |
|---|---:|---:|
| Observations receiving predictions | 208,417 / 208,417 | 19,160 / 19,160 |
| Eligible for point scoring | 134,669 (64.62%) | 13,126 (68.51%) |
| Nearest-road Top-1 accuracy on eligible points | 81,417 / 134,669 (60.457%) | 7,808 / 13,126 (59.485%) |
| Eligible reference present in first two | 97.35% | 97.32% |
| Eligible reference present in first four | 99.35% | 99.34% |
| Eligible reference present in first eight | 134,600 / 134,669 (99.949%) | 13,122 / 13,126 (99.970%) |
| Candidate misses | 69 | 4 |
| Trip-averaged Macro F1 | 0.4371 | 0.4542 |
| Continuous GPS steps, including stays | 82.78% | 82.02% |
| Directly connected steps among changes of arc | 26.24% | 28.92% |
| Trip-averaged ordered sequence F1 | 0.4588 | 0.4800 |
| GPS points requiring the nearest-road fallback | 18,230 | 0 |
| Geometry indexing, feature computation and nearest-road prediction, one local run | 5.63 s | 0.42 s |

The 10 student-test trips are for development. All point metrics use the approved Stage 2 derived labels. Eligibility favours observations close to a clearly separated reference arc, so the point scores cannot be extrapolated to the uncertain observations. The entire trip contributes to route metrics. No learned model was fitted in this stage.

## Prediction inputs and design decisions

The generator takes a GPS track, road nodes and directed arcs. It projects each trip into local metres with the Stage 2 function. The prediction path runs before the reference file is opened and has no argument for a route or label. Raw arc IDs identify roads within a trip and are excluded from the feature vector.

The shortlist keeps up to eight directed arcs within 50 m, ordered by point-to-segment distance. Equal computed distances are ordered by arc ID. Both directions of the same road share one canonical geometry calculation. A spatial index over samples spaced at most 40 m apart identifies segments for exact distance calculations. Every location on a segment is within 20 m of an indexed sample. If the radius contains no roads, the algorithm returns the globally nearest arc. The 50 m search and eight-candidate cap are Stage 3 design choices, fixed before student-test scoring. A fallback supplies an output even for a distant road and is preserved as an audit flag. Training trip 00000054 alone accounts for 16,519 of the 18,230 fallbacks.

Each candidate has eight numeric features. Distance to the road and excess distance over the best candidate measure geometric fit. Road length and projected position along the directed road describe local geometry. A heading cosine measures agreement between GPS movement and the road's travel direction. A five-second lookback supplies displacement-based speed and heading from earlier observations. A displacement below 3 m makes the heading unavailable. Two availability flags distinguish unavailable values from measured zero. The lookback cannot cross a gap longer than 30 seconds. A flag records whether the shortlist required a fallback. Candidate and label data are cached in separate files.

## How the baseline is scored

- Top-1 uses the first candidate, including its deterministic tie-break. Its denominator includes every eligible point, including candidate misses. For example, the four student-test misses count as incorrect.
- Macro F1 is computed independently per trip over the union of predicted and reference arc IDs on eligible points, then averaged equally over scorable trips. Top-1 and candidate coverage pool observation counts. The baseline outputs a rank and has no trained probability distribution, so its log loss is unavailable.
- A continuous step either stays on the same arc or changes to an arc starting at the previous arc's end node. GPS gaps over 30 seconds are excluded from adjacency checks. The change-only rate shows how often adjacent predictions connect once stays are removed. Skipping over an intermediate road can fail this strict check even when the vehicle could travel between the roads.
- Ordered sequence F1 is twice the longest common subsequence length divided by the sum of predicted and reference sequence lengths. Consecutive repeated predictions are collapsed within GPS runs, with fresh runs after gaps. The reference is compressed within runs of connected arcs. All runs remain in order for the sequence comparison. No intermediate road is inserted into a gap. This score measures ordered overlap of sampled predictions with the supplied route.
- Runtime covers spatial indexing, candidate generation, feature extraction and the nearest choice. It excludes file loading, reference alignment and scoring. It is an elapsed time from this machine.

## Two student-test errors to inspect

| Trip / zero-based GPS row | Evidence | Interpretation |
|---|---|---|
| 00000000 / 4 | Arc 212 ranks first at 0.27 m. Reference arc 17633 ranks second. | The directed arcs have reversed endpoints. Geometry alone ties, and the lower ID wins. |
| 00000000 / 1880 | Arc 3554 ranks first at 10.43 m. Reference arc 3691 is outside the first eight. | The reference satisfies the frozen 25 m label rule and therefore falls inside the 50 m search radius. The eight-road cap excludes it. |

The [error examples](error-examples.csv) give the first occurrence of each error type in each trip where it occurs. The [per-trip results](per-trip-results.csv) retain low-coverage trips, numerators, denominators and timing. The [baseline plot](baseline-results.png) compares shortlist depth and direction errors.

## Verification and reproduction

The index returned identical candidate IDs in the same order as an exhaustive search over all road arcs at 1,286 sampled real observations across all 90 trips. Separate checks covered reversed arcs, a segment crossing the GPS location with distant endpoints, zero-length roads, stationary movement, a long gap and fallback. All 80 training eligibility counts reproduced Stage 2. A separate calculation verified Top-1 counts and Macro F1 for every trip. The sequence routine matched a conventional dynamic program on 80 small cases.

With Python, NumPy 2.1.3 and SciPy 1.15.3, run the following in this folder. The optional cache stores feature arrays for later model comparisons. Each key in the compressed prediction archive combines the partition and trip ID, and its integer array is in GPS observation order. The plotting step uses Matplotlib 3.10.0.

```text
python run_stage3.py "path/to/Student folder.zip" --cache "path/to/local-cache"
python plot_baseline.py
```

**Provenance.** This stage used AI assistance to implement and run the baseline, verify the outputs and prepare this review. The review records measured development results for the team's inspection. It should be represented accordingly in any submission or disclosure.
