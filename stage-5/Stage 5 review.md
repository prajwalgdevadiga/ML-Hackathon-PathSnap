# PathSnap tuning and route reconstruction

**Recommendation for team review.** Use the tuned histogram gradient boosting classifier followed by HMM/Viterbi decoding for the two-laptop demonstration. On ten student-test trips, decoding raised sequence F1 from 0.6921 to 0.9068 and recovered 366 additional eligible point labels. Sequence F1 improved on every trip. The point score declined on one trip.

## Student-test results

| Method | Eligible points correct | Top-1 accuracy | Trip Macro F1 | Continuous steps | Connected changes of arc | Sequence F1 | Point log loss | Time across 10 trips |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Untuned histogram gradient boosting | 12,266 | 93.45% | 0.8790 | 91.40% | 69.12% | 0.6911 | 0.1858 | 0.226 s |
| Tuned histogram gradient boosting | 12,309 | 93.78% | 0.8814 | 90.90% | 67.12% | 0.6921 | 0.1868 | 0.437 s |
| Tuned classifier with Viterbi | 12,675 | 96.56% | 0.9621 | 97.70% | 86.66% | 0.9068 | N/A | 0.992 s |

The point-level denominator is 13,126 of 19,160 GPS observations (68.51%), using the approved Stage 2 alignment and eligibility rules. These are derived labels. Whole-trip sequence and continuity metrics use all predictions. The underlying candidate set is unchanged. Decoder output is a single most likely path, so classifier log loss is reported only for the first two rows. The table times begin with cached candidate features and exclude file I/O. A separate raw-input verification run, including candidate generation, took 1.385 s across all ten trips on this laptop. These are local compute measurements.

## Bayesian optimisation

The search maximised the mean trip-level Macro F1 on the same five training folds used in Stage 4. Each fold held out 16 complete trips. Each configuration was fitted five times. Three initial configurations included the Stage 4 baseline. Seven new configurations were proposed sequentially using a Matérn Gaussian process and expected improvement over 256 discrete possibilities. The optimiser only received scores from training folds. The seed remained 352, minimum samples per leaf remained 30, and early stopping remained off.

| Parameter | Values considered | Selected | Rationale |
|---|---|---:|---|
| Boosting iterations | 55, 75, 100, 135 | 135 | Balance model capacity and fitting cost |
| Maximum leaves per tree | 9, 15, 23, 31 | 31 | Test the complexity of interactions between geometry and movement |
| Learning rate | 0.05, 0.075, 0.10, 0.14 | 0.14 | Test the contribution of each new tree |
| L2 regularisation | 0.1, 1, 4, 12 | 4 | Control extreme leaf scores |

The tenth trial achieved the highest observed CV Macro F1, **0.85047**, up from **0.84585** for the untuned classifier. The search and final refit took 146.88 s, including 4.70 s for the final refit on all 80 training trips. This was a bounded search. The winner reached the upper tested values of three parameters. Using the same folds for optimisation and reporting makes the winning CV score a selection result. Student-test Macro F1 improved, though point log loss worsened slightly and the classifier took longer to score the ten trips. The [trial log](bayesian-trials.csv) records parameters, proposals, five fold scores and elapsed time for every trial.

## HMM and Viterbi specification

- **State and evidence.** A state is one directed candidate arc at one GPS observation. The ordered shortlist preserves the trip-local arc ID. The tuned classifier's normalised candidate odds provide discriminative emission weights. The initial prior is uniform over valid states within each run.
- **Transitions.** A path can stay on its directed arc, move onto an outgoing arc at the same junction, or traverse one intermediate directed arc. Positions projected onto both roads determine network travel distance. Its discrepancy from GPS displacement is measured in metres. Costs include excess backward movement on the same arc beyond 5 m, a 0.25 arc-change penalty, an additional 0.75 for a bridge and an additional 1.5 for a U-turn. Implausibly long transitions above the larger of GPS displacement plus 80 m or 60 m/s times the interval plus 30 m are excluded. The method can miss valid multi-road crossings between sparse observations.
- **Probabilities.** Each feasible transition has weight exp(−distance discrepancy / scale − penalties), including the backward-distance term in the discrepancy. Each row is normalised with an additional unobserved exit of weight exp(−3). The exit has zero emission at an observed road. Infeasible transitions have zero weight.
- **Decoding.** At each step Viterbi adds the current log emission to the best previous path score plus log transition probability and records the predecessor. It traces backward from the best final state. The best run ends at a GPS gap over 30 seconds or when no finite transition remains. A new run can then start at the current observation. No connecting road is invented across runs. An empty or entirely unsupported observation receives −1 and breaks the run. Stage 3's nearest-road fallback supplied at least one candidate at every observation in the measured trips.
- **Parameter selection.** Only held-out training emissions from the selected classifier were used to compare mismatch scales of 8, 20 and 45 m. Their mean trip sequence F1 scores were 0.8813, 0.8713 and 0.8651. The 8 m setting was frozen before scoring student-test trips. Other transition constants were engineering choices. The [decoder search](decoder-search.csv) includes point and continuity measures for all three settings.

## Where decoding helped and hurt

Decoding corrected 519 eligible classifier errors and introduced 153. Reverse-direction errors decreased from 453 to 264. Across all student GPS observations, the decoded paths used 417 single-arc bridges, restarted at two long gaps, and restarted 23 times when the allowed topology offered no continuing path. The continuity score compares successive observed arc assignments directly, so a valid one-road bridge still counts as a discontinuity under that metric. The sequence metric uses the predicted observation arcs with the frozen Stage 3 scoring convention. Inferred bridges are separately recorded and displayed.

| Student-test trip | Effect |
|---|---|
| 00000077 | 224 point errors corrected and 64 introduced, a net gain of 160 correct points. Sequence F1 increased by 0.5398. |
| 00000065 | 8 corrected and 36 introduced. Correct eligible points fell from 411 to 383 out of 420, while sequence F1 rose from 0.6519 to 0.9007. Greater agreement on the order of roads can coexist with disagreement on the road assigned to an individual observation. |

The [first student trip](route-00000000.html) and [point-accuracy regression](route-00000065.html) can be explored offline. The development maps display GPS points, the network, supplied reference, classifier route, decoded route and inferred connectors. Layer controls allow inspection of overlapping geometry. The reference layer is for development analysis only. The [screenshotted example](route-example.png), [comparison plot](stage-5-results.png), [per-trip results](trip-results.csv) and [summary](comparison.csv) are included.

## Reproduction and verification

The deployment-facing function `inference.predict_trip` accepts the fitted estimator, scale, GPS track, nodes and arcs. It returns raw classifier assignments, decoded assignments, run starts, connecting arc IDs and fallback flags. The reference route is read by the scoring and map scripts. The ten raw-input replays reproduced every student-test prediction, connector and run boundary from the saved model and configuration. Graph checks validated every transition inside a run. An independent F1 calculation verified each decoded trip's point score. On 30 small problems Viterbi matched exhaustive path enumeration. Separate cases exercised disconnected candidates, empty observations and a long gap.

The files use the Stage 4 Python environment (Python 3.13.5, NumPy 2.1.3, SciPy 1.15.3, scikit-learn 1.6.1 and joblib 1.4.2). Map and plot generation uses Folium 0.19.7, Branca 0.8.1 and Matplotlib 3.10.0. From this folder, reproduce the trial log, decoder selection, predictions and plots with the commands below. The first command is needed only to rebuild the Stage 3 cache. The scratch directory stores the best trial's out-of-fold emissions.

```text
python shared/stage3/run_stage3.py "path/to/Student folder.zip" --out "path/to/baseline" --cache "path/to/cache"
python tune.py "path/to/Student folder.zip" "path/to/cache" "path/to/scratch"
python evaluate.py "path/to/Student folder.zip" "path/to/cache" "path/to/scratch"
python verify.py "path/to/Student folder.zip" "path/to/cache"
python visuals.py "path/to/Student folder.zip"
```

**Provenance.** This checkpoint records AI-assisted implementation, execution and analysis. The team can reproduce the measurements and review the assumptions against its own understanding before using the material in a submission.
