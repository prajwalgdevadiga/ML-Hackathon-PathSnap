# PathSnap two-laptop rehearsal

Laptop 1 runs FastAPI, the fitted histogram gradient boosting classifier, the Viterbi decoder, and private reference scoring. Laptop 2 opens the evaluator interface in a browser, selects a trip, and sees the predicted route and aggregate scores. All interface assets, including Leaflet, are bundled for offline use.

## Run on Laptop 1

Extract the entire package. Open Command Prompt in the extracted folder. These commands use Python 3.13, the version used for the verified model. Replace the ZIP path and the address with the actual locations. Use the Wi-Fi IPv4 address shown by Windows `ipconfig`.

```bat
py -3.13 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe server\service.py --data "C:\path\Student folder.zip" --partition student_test --mode student-test --host YOUR_WIFI_IPV4 --port 8765 --log receiving.log
```

Keep this Command Prompt open during the demo. The server prints a request receipt containing the request ID, requesting device address, trip ID, correct and eligible counts, and inference time. The HTTP access log also records the endpoint and response status. To stop the server, press Ctrl+C in its Command Prompt.

## Use Laptop 2

1. Connect both laptops to a network that permits direct connections between devices.
2. In Laptop 2's browser, open `http://YOUR_WIFI_IPV4:8765/`, using Laptop 1's address. Wait for **API connected** and verify there are ten trip options.
3. Select a trip, click **Reconstruct route**, and compare the GPS and predicted-route layers. The displayed point accuracy uses only the scorable GPS observations. The page shows both the denominator and the total.
4. Match the request ID under the scores to the `RESULT` line in Laptop 1's Command Prompt. Capture the working UI on Laptop 2 and the receiving Command Prompt on Laptop 1 for the report. Repeat for any of the ten trips.

If Laptop 1 can open the address and Laptop 2 times out, verify that both devices are on the same reachable network. A campus or guest Wi-Fi can isolate devices. A shared hotspot that permits client-to-client connections is another way to rehearse. The listening address must be updated if Laptop 1's Wi-Fi address changes.

## Reserved evaluation trips

Use the organiser-provided ten trips only at evaluation time on Laptop 1. For a ZIP, set `--partition` to the exact parent folder of the ten trip folders. For a directory, pass the partition directory itself to `--data`. Each trip folder must contain its matching `.track`, `.nodes` and `.arcs` files. By default the scoring step reads the corresponding `.route` file. If organisers provide reference files separately, use `--references` to point to their private folder. Both `TRIP.route` and `TRIP/TRIP.route` are supported there.

```bat
.venv\Scripts\python.exe server\service.py --data "C:\event\reserved-trips.zip" --partition evaluator_test --mode evaluator --references "C:\event\private-reference" --host YOUR_WIFI_IPV4 --port 8765 --log receiving.log
```

The active source determines the ten trip IDs. The `--mode` setting controls the display label. Keep the private dataset outside `web/assets`, the static directory served to browsers. Both the development and evaluation modes perform inference for each POST. If an official evaluator scoring procedure is supplied, it should replace the current server-side derived-label scoring step.

## Interface and verification

| Request | Response |
|---|---|
| GET `/` | Browser interface |
| GET `/api/trips` | Active partition label and ten available IDs |
| POST `/api/trips/{id}/predict` | Request ID, predicted arc per observation, predicted route geometry, GPS, road context, inference time and aggregate metrics |

Each inference uses only GPS and road-network arrays. Once the prediction is complete, server scoring reads the private reference. The response includes predicted arc IDs and public input geometry. Reference IDs, reference routes and per-point eligibility or correctness flags stay on Laptop 1. A trip ID outside the active partition returns HTTP 404. A missing scoring reference returns HTTP 503.

The [ten-trip API check](evidence/api-checks.csv) records the live development run. The [working UI screenshot](evidence/working-ui.png) was captured in a browser on Laptop 1. The saved models and Stage 5 source are the basis of the bundled inference code. To reproduce the all-trip comparison against the separate Stage 5 package, run the following while the rehearsal API is online.

```bat
.venv\Scripts\python.exe verify_development.py http://YOUR_WIFI_IPV4:8765 "C:\path\stage-5"
```

**Provenance.** This package was implemented and tested with AI assistance. Its evidence comes from actual runs on the supplied student-test partition. The reserved evaluator data were not used.
