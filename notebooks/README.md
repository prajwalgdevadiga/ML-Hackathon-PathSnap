# Run the PathSnap training notebooks

Extract the full PathSnap project ZIP. Keep these notebooks in the project’s `notebooks` folder and the original student dataset at `inputs/Student folder.zip`. Open either notebook in Jupyter or VS Code and select a Python 3.13 kernel. Each notebook has an installation cell for the training dependencies. Run each notebook from top to bottom.

| Notebook | What it runs |
|---|---|
| [01_train_and_compare.ipynb](01_train_and_compare.ipynb) | Fits logistic regression, Gaussian Naive Bayes, a decision tree, a random forest and histogram gradient boosting using five trip-disjoint folds. It also runs the logistic regression heading ablation. |
| [02_bayesian_tuning_and_viterbi.ipynb](02_bayesian_tuning_and_viterbi.ipynb) | Runs ten Bayesian trials, refits histogram gradient boosting and compares classifier-only and Viterbi-decoded results. |

Either notebook can run first. Run them one at a time to reuse the feature cache. If you downloaded a notebook separately, place it in the extracted project’s `notebooks` folder or set `PROJECT_ROOT` and `DATASET_ZIP` in its setup cell.

Each run takes several minutes. The last cell reads the accuracy table from that run. The run directory printed below the table contains the detailed results, saved models and verification outputs. For comparison, the original tuned model scored **93.78% Top-1** on scorable student-test observations. It scored **96.56%** after Viterbi decoding.
