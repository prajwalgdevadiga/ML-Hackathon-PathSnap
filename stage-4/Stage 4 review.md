# PathSnap Stage 4 model comparison

**Recommendation for team review.** Take histogram gradient boosting into Bayesian tuning. It ranked first on the primary measure, the mean trip-level out-of-fold Macro F1 across all 80 training trips. Its score was 0.8458, compared with 0.8395 for random forest. The primary measure gives each trip and its road classes weight. Student-test results are development comparisons. No evaluator data were available or used.

| Algorithm | CV Macro F1 | Student Top-1 | Student Macro F1 | Student log loss | Continuity | Sequence F1 | Inference, 10 trips |
|---|---:|---:|---:|---:|---:|---:|---:|
| Logistic regression | 0.8091 | 92.84% | 0.8414 | 0.2870 | 90.48% | 0.6834 | 0.019 s |
| Gaussian Naive Bayes | 0.8060 | 92.77% | 0.8337 | 0.5062 | 90.54% | 0.6823 | 0.031 s |
| Decision tree | 0.8351 | 93.15% | 0.8602 | 0.2842 | 90.03% | 0.6638 | 0.013 s |
| Random forest | 0.8395 | 93.55% | 0.8712 | 0.1818 | 91.06% | 0.6892 | 0.248 s |
| Histogram gradient boosting | 0.8458 | 93.45% | 0.8790 | 0.1858 | 91.40% | 0.6911 | 0.240 s |

## Travel direction and baseline

The paired logistic-regression experiment dropped heading cosine and its availability flag while retaining speed. Student-test Top-1 was 59.89% without those features and 92.84% with them, a difference of 32.95 percentage points. Reverse-direction mistakes changed from 4,688 to 507. Out-of-fold Macro F1 changed from 0.4192 to 0.8091. The earlier nearest-road baseline achieved 59.485% student-test Top-1, Macro F1 of 0.4542 and sequence F1 of 0.4800. It made 4,741 reverse-direction errors.

## Design and scoring

All five classifiers take the eight frozen features from Stage 3 and estimate whether a candidate is the derived target for an observation. Each eligible candidate row is one binary training example. All available candidates are included with equal row weight, including all-negative sets when the reference is missing. The full training fit uses 709,934 candidate rows and 134,600 positive rows. Uncertain observations are retained at prediction time. Trip IDs, arc IDs and reference-derived quantities are excluded from model features.

Five folds are formed by shuffling the 80 unique training trip IDs with seed 352, then assigning 16 entire trips to validation per fold. Every training trip is scored once out of fold. Each algorithm is also refitted on all 80 trips for the 10-trip student-test comparison. The logistic heading check is a controlled variant of the same algorithm. Configuration, fold membership, fit times, per-trip scores and predictions are included in the package.

At prediction time, positive-class probabilities are clipped to [1e-12, 1 − 1e-12] and converted to odds. Odds are normalised across valid candidates to produce a trip-local candidate distribution per point. Equal scores prefer the fixed Stage 3 candidate order. The probabilities are estimates for the shortlist. They have not undergone a separate calibration fit. Point log loss is mean negative log probability of the eligible reference with a floor of 1e-12, including candidate misses.

Point metrics use the frozen derived labels. Training has 134,669 eligible points out of 208,417 and student testing has 13,126 out of 19,160. Four student-test candidate misses and 69 training misses count against every algorithm. Pooled Top-1 weights observations. Macro F1, precision and recall average over the union of predicted and reference arcs within each trip, then weight trips equally. A training trip with only one eligible observation has the same trip weight and increases fold variability.

Route continuity uses every prediction and excludes intervals exceeding 30 seconds. It counts a stay or a directly connected arc change as continuous. Sequence F1 compares ordered predicted and reference arc lists using the Stage 3 longest-common-subsequence protocol. Change-only continuity is available in the CSV. Inference times cover candidate scoring and ranking on cached features, including probability checks. They exclude candidate generation, reference alignment, metrics and API transport. All timings are single local measurements.

## Reading the individual results

### Logistic regression

The linear model estimates candidate log odds with L2-regularised logistic loss. Standardisation is fitted inside each fold. C = 1 and the iteration limit is 250. This gives a fast, interpretable test of distance and direction, with limited ability to express interactions.

Full training took 0.64 s. The refit scored 0.8098 Macro F1 on its training trips and 0.8091 out of fold. Fold scores ranged from 0.7754 to 0.8353. The observed train-minus-validation difference was +0.0007. On student testing it got 12,186 of 13,126 eligible points correct, with macro precision 0.8599 and macro recall 0.8322.

### Gaussian Naive Bayes

The generative model estimates class priors and class-conditional Gaussian means and variances for standardised features, with variance smoothing of 1e-8. It is inexpensive to fit. Correlation between distance and distance above the nearest candidate, and the binary availability flags, challenge its assumptions.

Full training took 0.16 s. The refit scored 0.8076 Macro F1 on its training trips and 0.8060 out of fold. Fold scores ranged from 0.7669 to 0.8312. The observed train-minus-validation difference was +0.0016. On student testing it got 12,177 of 13,126 eligible points correct, with macro precision 0.8522 and macro recall 0.8241.

### Decision tree

A single decision tree uses Gini impurity, maximum depth 12 and at least 30 candidate rows per leaf. It can express different rules for moving and stationary observations. Its probability estimates depend on individual leaf counts.

Full training took 5.24 s. The refit scored 0.8495 Macro F1 on its training trips and 0.8351 out of fold. Fold scores ranged from 0.8141 to 0.8516. The observed train-minus-validation difference was +0.0144. On student testing it got 12,227 of 13,126 eligible points correct, with macro precision 0.8764 and macro recall 0.8524.

### Random forest

The forest averages positive-class probabilities from 48 Gini trees. Each uses a half-size bootstrap sample, square-root feature sampling, maximum depth 14 and at least 12 rows per leaf. Averaging can stabilise a single tree's rules, at a cost in prediction time.

Full training took 19.01 s. The refit scored 0.8509 Macro F1 on its training trips and 0.8395 out of fold. Fold scores ranged from 0.8181 to 0.8530. The observed train-minus-validation difference was +0.0114. On student testing it got 12,280 of 13,126 eligible points correct, with macro precision 0.8866 and macro recall 0.8637.

### Histogram gradient boosting

The histogram-based ensemble fits additive trees to binary log loss. The fixed configuration uses 75 iterations, learning rate 0.1, up to 15 leaves, at least 30 rows per leaf and L2 penalty 1. Early stopping is disabled. It can learn how speed and direction interact, with more fitting choices than a linear model.

Full training took 3.02 s. The refit scored 0.8475 Macro F1 on its training trips and 0.8458 out of fold. Fold scores ranged from 0.8273 to 0.8557. The observed train-minus-validation difference was +0.0017. On student testing it got 12,266 of 13,126 eligible points correct, with macro precision 0.8943 and macro recall 0.8715.

The refit–validation differences describe this dataset and are not formal tests of overfitting. Variation across trips and ambiguous point labels limits fine-grained claims about algorithm rank. No epoch losses were recorded for these runs.

## Reproduction and provenance

Use Python 3.13.5, scikit-learn 1.6.1, NumPy 2.1.3, SciPy 1.15.3, joblib 1.4.2 and Matplotlib 3.10.0. From this directory, the first command rebuilds the approved Stage 3 cache if needed. Place generated intermediates in a local working directory.

```text
python stage3/run_stage3.py "path/to/Student folder.zip" --out "path/to/baseline" --cache "path/to/cache"
python compare.py "path/to/Student folder.zip" "path/to/cache"
python verify.py "path/to/cache"
python review.py
```

This review records AI-assisted implementation and execution. It is a development checkpoint for team review. Saved estimators, predictions, configurations and measured outputs support independent reproduction and the team's own interpretation. Hyperparameter tuning and route decoding follow a separate approval checkpoint.
