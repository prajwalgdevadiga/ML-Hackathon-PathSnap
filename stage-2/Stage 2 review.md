# PathSnap training-data review

**Decision for the team.** Use the ordered reference route to derive provisional point labels. Report the fraction of GPS observations that receive a usable label alongside any point-level score. On the training partition the proposed rule retains 134,669 of 208,417 points (64.62%). The archive supplies no independent point labels against which to measure alignment accuracy.

## What was measured

The analysis reads only the 80 training trips in the supplied ZIP. Across their GPS tracks and node tables there are zero missing numeric cells and zero repeated GPS timestamps. GPS coordinates span longitudes −100.418538° to 137.237029° and latitudes −34.640291° to 61.267258° across the trips. A local tangent-plane approximation around each trajectory's mean coordinate converts degree differences to metres using WGS84 latitude-dependent scale factors. Each arc is represented by the straight segment between its two supplied nodes. Distances therefore inherit any limitations of this geometry.

| Check | Observation | Practical consequence |
|---|---|---|
| Sampling | 201,048 of 208,337 intervals last one second. There are 133 gaps longer than 30 seconds across 28 trips. | Split movement features and decoder runs at long gaps. |
| Road connectivity | In 79 of 80 trip graphs, some nodes cannot reach one another even if road direction is ignored. Graphs contain 1–421 such components. There are zero isolated nodes. | A route between successive candidate roads can be impossible. |
| Route continuity | 13 consecutive reference-arc pairs across 10 trips do not share the expected directed end/start node. | Flag these boundaries and avoid assuming every supplied reference transition is valid. |
| GPS proximity | Median distance to the order-aligned segment is 5.67 m. The 95th percentile is 146.40 m. In 26,849 cases every arc in the supplied reference is more than 25 m away. | A large offset can persist even if the order requirement is removed. |

The [trip inventory](training-trip-audit.csv) gives bounds, graph counts, quality flags and coverage for every training trip. The [distribution plots](training-distributions.png) show all training GPS intervals and alignment distances.

## How point labels were derived

For each observation, compute its distance to every segment in that trip's reference sequence. Use dynamic programming to select a sequence of reference positions that stays in place or moves forward and minimises the total distance, with each point's cost capped at 60 m. The cap limits the influence of distant observations. Raw distances are used for all quality checks. The path may skip unsampled arcs and may start or finish inside the sequence. Tie-breaking favours the earliest reference position. Timing and speed do not enter this alignment, and it does not infer what happened during a gap. This offline use of the complete trip and its reference is solely for deriving labels.

For each assigned segment, compute the distance to the closest **different arc ID in the reference** and subtract the distance to the assigned segment. A small margin indicates competing geometry. Repeated occurrences of the same arc ID do not count as competing labels. A spatial tie between opposite directed arcs remains ambiguous under this rule even when travel direction could resolve it.

Proposed eligibility requires all three conditions.

1. The GPS observation lies within 25 m of the assigned reference arc.
2. Every other distinct reference arc is at least 3 m farther away.
3. The observation is clear of either endpoint of a GPS gap over 30 seconds or an aligned step that crosses a break in the directed reference sequence.

Among 208,417 GPS points, 27,231 exceed the distance threshold, 62,231 fail the margin threshold, and 268 touch a marked boundary. These counts overlap. Altogether 73,748 points are uncertain. If route order is ignored, the nearest reference arc disagrees with the order-aligned arc on 10,377 points (4.98%), including distance ties. This disagreement is a diagnostic and cannot establish which assignment is physically correct.

| Maximum distance / minimum margin | Usable points | Share of all training GPS |
|---|---:|---:|
| 15 m / 5 m | 118,575 | 56.89% |
| **25 m / 3 m** | **134,669** | **64.62%** |
| 40 m / 1 m | 151,698 | 72.79% |

All three settings use the same boundary exclusions. The middle setting is a proposed working choice. The sensitivity comparison measures how much data the rule retains. It does not estimate the error rate of the resulting labels.

## Inspect two trips

The [representative trip map](map-00000083.html) uses the training trip closest to the median observation count, breaking ties by trip ID. All 1,507 GPS points are shown, of which 1,168 meet the proposed rule. Toggle the layers to inspect the road network, reference and uncertain observations. Hover over a point to see its assigned arc and distances. The [point-by-point alignment](alignment-00000083.csv) includes the following transition. Indices start at zero.

| GPS row | Time (s) | Assigned arc | Distance (m) | Other-arc margin (m) | Decision |
|---:|---:|---:|---:|---:|---|
| 42 | 43 | 7175 | 21.86 | 6.27 | Retain |
| 43 | 44 | 7175 | 25.32 | 2.44 | Uncertain |
| 44 | 45 | 4779 | 12.62 | 13.53 | Retain |

The [long-gap trip map](map-00000090.html) shows 536 points, of which 275 meet the rule. The last two points are separated by 153,064 seconds. Both are marked uncertain, even though neither is far from its assigned road. The [long-gap alignment](alignment-00000090.csv) records their timestamps as 970 and 154,034 seconds.

Two further cases explain why aggregate coverage is insufficient. In trip 00000011, the reverse edge of 564 of its 565 route entries also appears in the reference. Only one of 2,828 points passes the rule. Trip 00000054 has a different problem. For 17,167 of 19,313 points, even the nearest reference arc is over 25 m away. Both trips should stay visible in evaluation.

## Proposed scoring agreement

- Freeze the label and eligibility rules using training data before scoring student-test trips. Keep reference files, derived labels, alignment distances and eligibility flags out of prediction inputs.
- Produce predictions for all relevant observations, including uncertain ones. Report point-level Top-1 accuracy and Macro F1 against the eligible derived labels, with coverage and the number of eligible points per trip. Use trip-local arc IDs for F1 and state the averaging convention. A trip with no eligible points receives an explicit unavailable point score.
- Count a missing reference candidate as an incorrect Top-1 prediction. Report candidate miss rate separately. Coverage records uncertainty in the **reference label** and must be computed independently of the candidate generator or model.
- Report route continuity and an ordered sequence comparison on complete trips, with explicit handling of gaps and invalid reference transitions. Keep difficult and low-coverage trips visible. The retained point set favours geometrically clear roads, which can favour a distance-based baseline.
- Use the same frozen protocol for every model and decoder. If organisers provide an official point-alignment or scoring procedure, use it for official comparisons and document any differences.

## Reproduction and checks

Run `python explore.py "path/to/Student folder.zip"` from this folder. The inspected run used NumPy 2.1.3, SciPy 1.15.3, Matplotlib 3.10.0, Folium 0.19.7, Branca 0.8.1 and xyzservices 2025.4.0. The two maps embed Leaflet 1.9.4 from the adjacent assets directory and display the supplied network offline. Its license is included in that directory.

The alignment routine matched exhaustive enumeration on 40 small test cases, and the distance calculation matched analytical examples including a zero-length segment. The two maps were checked in a browser. These checks verify the implementation and display. The team still needs to review the assumptions and interpretations before adopting them for experiments or reporting them as its own analysis.
