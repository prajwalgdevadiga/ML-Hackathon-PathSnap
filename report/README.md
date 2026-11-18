# Report and demonstration evidence

Open [PathSnap Technical Report](PathSnap-Technical-Report.pdf). Pages 12–16 show all ten student-test trips from the deployed server. Page 11 lists the request IDs and scores. The individual screenshots, [capture manifest](figures/student-test/manifest.csv) and [matching server log](figures/student-test/server-requests.log) are under `figures/student-test/`. Screenshots are browser captures of live requests. The older two-device rehearsal is documented in [stage-6 evidence](../stage-6/evidence/).

## Rebuild the PDF

From the repository root after setting up Python 3.13:

```powershell
.\.venv\Scripts\python.exe -m pip install -r .\report\requirements.txt
.\.venv\Scripts\python.exe .\report\build_report.py
```

The report uses the saved classroom experiment tables and included PNGs. To regenerate the screenshots, start a local student-test server with `.\start-server.cmd student-test 127.0.0.1`. Use a separate command window to run the capture script with Microsoft Edge installed. This replaces the screenshots and manifest with fresh request IDs. Copy the log from that same local run, then rebuild the PDF.

```powershell
.\.venv\Scripts\python.exe .\report\capture_trip_screenshots.py http://127.0.0.1:8765/
Copy-Item .\receiving.log .\report\figures\student-test\server-requests.log
.\.venv\Scripts\python.exe .\report\build_report.py
```
