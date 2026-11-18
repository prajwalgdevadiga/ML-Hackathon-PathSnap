# PathSnap evaluator guide

The rubric has five categories worth five marks each. Use the evidence below in rubric order. The GitHub package includes code, results and report evidence. The input archives are supplied separately. See the [dataset setup](inputs/README.md).

| Rubric item | What to show | Files to open |
|---|---|---|
| **Data Preprocessing and Feature Extraction (5)** | Explain how the GPS tracks, road graph and ordered reference routes are read. Show how the pipeline aligns the reference, flags uncertain point labels, and generates up to eight directed roads per point. Describe the eight features and why heading helps distinguish opposite directions. | [Input audit](stage-1/Input%20audit.md), [exploration](stage-2/training-trip-audit.csv), [training-trip map](stage-2/map-00000083.html), [reference alignment](stage-3/reference.py), [feature extraction](stage-3/candidates.py), [baseline results](stage-3/per-trip-results.csv) |
| **Model Selection and Training (5)** | Show the five actual fits, the fixed trip-disjoint folds, and the student-test comparison. Histogram gradient boosting led validation Macro F1 at **0.8458**. Random forest led untuned student-test Top-1 at **93.55%**. The logistic regression heading ablation is an additional experiment. | [Training notebook](notebooks/01_train_and_compare.ipynb), [training script](stage-4/compare.py), [folds](stage-4/folds.json), [results](stage-4/comparison.csv), [fit log](stage-4/fit-log.csv), [comparison figure](stage-4/model-comparison.png) |
| **Hyperparameter Tuning (5)** | Explain the ten Bayesian trials and validation objective. Show the winning values: 135 iterations, 31 leaves, learning rate 0.14 and L2 regularisation 4. Validation Macro F1 rose from **0.84585** to **0.85047**. Student-test point accuracy rose from **93.45%** to **93.78%**. | [Tuning notebook](notebooks/02_bayesian_tuning_and_viterbi.ipynb), [Bayesian search code](stage-5/tune.py), [trial history](stage-5/bayesian-trials.csv), [selected parameters](stage-5/tuning.json), [before/after scores](stage-5/comparison.csv) |
| **ML Ops (5)** | Show the saved estimator, decoder configuration, pinned dependencies, and the FastAPI request path. The UI sends a trip ID. The server loads GPS and geometry, predicts and decodes, then reads private reference data for scoring. Show the results of the 10-trip API check and match a live request ID to the server's receiving log. | [Deployment guide](stage-6/README.md), [dependencies](stage-6/requirements.txt), [API](stage-6/server/service.py), [fitted model](stage-6/server/tuned.joblib), [Viterbi code](stage-6/server/decoder.py), [decoder settings](stage-6/server/decoder-config.json), [API verification](stage-6/evidence/api-checks.csv), [receiving log](stage-6/evidence/receiving-rehearsal.log) |
| **Demo and Deliverables (5)** | Open the UI on Laptop 2, select a trip, and show the world map zooming to the predicted route. Match the Request ID in Laptop 1's Command Prompt. Present the consolidated PDF and public GitHub repository. | [UI files](stage-6/web/index.html), [world map screenshot](report/figures/world-map.png), [route UI screenshot](report/figures/working-ui.png), [technical report](report/PathSnap-Technical-Report.pdf), [interactive Folium map](stage-5/route-00000065.html) |

## Live demo

The revised report contains screenshots of all ten student-test runs. Use the [capture manifest](report/figures/student-test/manifest.csv) and [server request log](report/figures/student-test/server-requests.log) to confirm their request IDs.

1. Start FastAPI on Laptop 1 using the [deployment guide](stage-6/README.md). Open its Wi-Fi address on Laptop 2. Use student-test mode for rehearsal and use the organiser-provided reserved trips only at evaluation time.
2. Select trip 00000077 for rehearsal. The recorded second-device request returned **2,123 of 2,331 eligible points correct**. The browser displays the route and scores and generates a new request ID on each run.
3. Match that ID to Laptop 1's successful POST and RESULT lines. Capture the receiving Command Prompt and the result screen for the submission evidence.
4. If asked to rerun training, open the notebooks in a Python 3.13 kernel and use **Run All**. Both have been tested with the packaged dataset. Run them one at a time.

## Numbers to explain

- Tuned classifier only: **12,309 / 13,126 = 93.78%** eligible point accuracy.
- With HMM and Viterbi: **12,675 / 13,126 = 96.56%** eligible point accuracy. Sequence F1 improved from **0.6921** to **0.9068**. These results are from the 10 student-test trips.
- **68.51%** of student-test observations were scorable under the reference-label rule. The model produced predictions for all 19,160 observations.
- On trip 00000065, sequence F1 improved from 0.6519 to 0.9007 while correct eligible point assignments fell from 411 to 383. Route order and point alignment measure different outcomes.

## Submission status

The report and project files are ready for team review. The log records successful requests from a second device. The receiving Command Prompt screenshot is still needed for the report. The local repository is prepared for publication under the team's chosen GitHub account. Add the public URL after publication. Fill the team names, SRNs and section on the evaluator's form.
