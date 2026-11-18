# Run PathSnap on another Windows laptop

Install Python 3.13. Extract the project ZIP and open Command Prompt inside the extracted `PathSnap` folder. Run this setup once on the laptop that will host the server.

```bat
py -3.13 -m venv .venv
.venv\Scripts\python.exe -m pip install -r stage-6\requirements.txt
```

## Student-test rehearsal

```bat
start-server.cmd student-test
```

This uses the bundled `inputs\Student folder.zip` on port 8765. After setup, double-clicking `start-server.cmd` also starts this mode.

## Reserved evaluation

When the evaluator authorizes the run, copy their original `evaluator_test.zip` into the extracted project's `inputs` folder. Keep it zipped. Start the server with the saved model.

```bat
start-server.cmd evaluator
```

This uses port 8766. The reserved ZIP is supplied separately by the evaluator.

## Connect from Laptop 2

1. Keep Laptop 1's server Command Prompt open. In a second Command Prompt on Laptop 1, run `ipconfig` and find the Wi-Fi IPv4 address.
2. On Laptop 2 on the same Wi-Fi, open `http://LAPTOP_1_IP:8765/` for rehearsal or `http://LAPTOP_1_IP:8766/` for evaluator mode. Replace `LAPTOP_1_IP` with that address.
3. Select a trip and click **Reconstruct route**. Match the displayed request ID to Laptop 1's Command Prompt. A copy is written to `receiving.log`.

For a same-laptop rehearsal, run `start-server.cmd student-test 127.0.0.1`, then open `http://127.0.0.1:8765/`. Use Ctrl+C in the receiving Command Prompt to stop the server.

## Run the notebooks

Use Python 3.13 in Jupyter or VS Code. Open the two notebooks under `notebooks` and run one at a time. The first trains and compares the five classifiers. The second reruns Bayesian tuning and evaluates Viterbi. The expanded search code and measured results are under `stage-5`. See [the evaluator guide](EVALUATOR-GUIDE.md) for the full evidence index.
