# PathSnap

PathSnap reconstructs vehicle routes from noisy GPS observations. A histogram gradient boosting classifier ranks nearby directed roads. An HMM with Viterbi decoding chooses a sequence of roads that is consistent with the observed movement and the road network.

The deployed pipeline matched 12,675 of 13,126 eligible GPS points (96.56%) across the ten student-test trips. Mean trip sequence F1 was 0.9068. The [technical report](report/PathSnap-Technical-Report.pdf) includes ten live screenshots, results from the separate expanded search, and the limits of point-level scoring. The [evaluator guide](EVALUATOR-GUIDE.md) lists the evidence by rubric item.

## Run locally

Use Python 3.13. Follow the [input instructions](inputs/README.md) to place the organiser-supplied `Student folder.zip` in the project. Open PowerShell at the repository root.

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r .\stage-6\requirements.txt
.\start-server.cmd student-test 127.0.0.1
```

Open http://127.0.0.1:8765/ in a browser, select a trip and click **Reconstruct route**. For the two-laptop demo and the reserved evaluator mode, follow the [Windows runbook](RUN-ON-ANOTHER-LAPTOP.md). The server loads the saved checkpoint.

## Reproduce the experiments

The [training notebooks](notebooks/) run the same code that produced the original results. Open them in Jupyter or VS Code from the extracted project and run each notebook from top to bottom. The first compares the five classifiers. The second reruns Bayesian tuning and measures the effect of Viterbi decoding. Both print accuracy tables from their fresh runs and save the detailed results under `work/notebook-runs/`.

The following folders contain the code and evidence in execution order. Python library versions and commands are included in the stage reviews.

| Folder | Work |
|---|---|
| [stage-1](stage-1) | Validate the raw archive and inventory all 90 trips |
| [stage-2](stage-2) | Explore training trips, verify connectivity and derive point labels |
| [stage-3](stage-3) | Build candidate features, generate the reusable cache and run the nearest-road baseline |
| [stage-4](stage-4) | Compare five classifiers on identical trip folds, including an additional heading ablation |
| [stage-5](stage-5) | Reproduce the Bayesian search, tune the decoder, compare routes and generate Folium maps |
| [stage-6](stage-6) | Run the fitted pipeline, server-side scoring, browser interface and all-trip API verification |
| [report](report) | Technical report, ten browser screenshots, matching request log and build instructions |

Outputs under each stage are from actual development runs. AI assistance was used in project implementation, execution, verification and report drafting. The team is responsible for reviewing that assistance and its technical conclusions under the event policy. Physical two-laptop evidence is tracked separately from local rehearsal evidence.
