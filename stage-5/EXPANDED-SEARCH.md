# Expanded tuning and generalization check

This experiment compared more configurations without opening the reserved evaluator ZIP. Candidate features, trip folds and reference-label rules stayed fixed. The search was staged so that classifier and decoder settings could be evaluated separately.

| Phase | Settings covered across the original and expanded searches | Additional configurations |
|---|---|---:|
| Histogram gradient boosting | Iterations, maximum leaves, learning rate, L2 regularization, maximum depth, minimum samples per leaf, histogram bins, feature subsampling | 10 |
| Route decoder | Displacement scale, backward tolerance, arc-change penalty, bridge penalty, reverse-arc penalty, gap timeout, exit log weight, emission temperature, distance-screen speed cap, GPS-distance slack, speed-distance slack, one-road bridge allowance | 17 |

The pipeline covers **20 distinct settings**. Four classifier settings and the decoder scale were already tested in the original experiments. The expanded search added 15 settings. Ten more classifier configurations and 17 decoder configurations were scored, alongside a replay of the current defaults at each stage. The added configurations form a targeted search. The original ten-trial Bayesian search has its own record.

The classifier selection minimizes point log loss among configurations within 0.002 Macro F1 of the best training-fold result. This places a bound on the loss in arc ranking while allowing an improvement in the probabilities used by Viterbi. The decoder then maximizes mean trip sequence F1 on held-out training predictions while allowing a decrease of at most 0.10 percentage points in point accuracy versus the original decoder settings on the selected classifier's emissions. The selected settings are saved before student-test scoring.

Run the following from the project root with the Python dependencies from the training notebooks installed. Skip the first command if you already have the candidate cache and point the second command to it.

```text
python stage-3/run_stage3.py "inputs/Student folder.zip" --out work/baseline --cache work/cache
python stage-5/expanded_search.py "inputs/Student folder.zip" work/cache work/expanded-results work/expanded-scratch
```

The [classifier trials](expanded/classifier-trials.csv), [decoder trials](expanded/decoder-trials.csv), [frozen selection](expanded/selection.json) and [student-test comparison](expanded/student-comparison.csv) record the resulting measurements. The candidate estimator is saved separately from the deployed model. The repeated searches use the same folds, so these are development comparisons. The reserved evaluator partition is still required for an independent final check.

## Student-test results

| Pipeline | Top-1 accuracy | Correct / eligible | Macro F1 | Sequence F1 | Continuity |
|---|---:|---:|---:|---:|---:|
| Current classifier | 93.78% | 12,309 / 13,126 | 0.8814 | 0.6921 | 90.90% |
| Current Viterbi | 96.56% | 12,675 / 13,126 | 0.9621 | 0.9068 | 97.70% |
| Expanded classifier | 93.64% | 12,291 / 13,126 | 0.8830 | 0.6933 | 91.06% |
| Expanded Viterbi | 96.76% | 12,701 / 13,126 | 0.9682 | 0.9167 | 97.82% |

The expanded configuration used depth 5 in addition to the prior classifier settings. Held-out classifier Macro F1 was 0.84997 versus 0.85047. Held-out log loss improved from 0.23461 to 0.22584. Its in-sample Top-1 was 94.38% versus approximately 92.41% held out. The current classifier's corresponding values were 94.93% and 92.41%. The reduced gap comes mainly from lower in-sample accuracy.

The selected decoder used a 6 m scale, 3 m backward tolerance, a 0.4 arc-change penalty, a 1.0 bridge penalty, a 2.0 reverse penalty, a 25 s gap timeout, an exit log weight of -3.5 and an emission temperature of 1.15. On the selected classifier's held-out training emissions, the decoder raised sequence F1 from 0.8822 to 0.8900 and point accuracy from 96.79% to 96.95%.

On the student-test set the expanded pipeline recovered 26 more eligible points than the current deployed pipeline. The expanded classifier by itself recovered 18 fewer. The baseline row was reproduced with the existing model and decoder settings. The live evaluator continues to use its previously deployed checkpoint.
