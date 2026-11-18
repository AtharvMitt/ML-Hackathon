# PathSnap

PathSnap matches a GPS track to the directed roads in its supplied network. Laptop 1 runs a candidate classifier, a Viterbi decoder and private scoring. A browser on Laptop 2 selects a trip and displays the reconstructed route.

## Current results

The saved model is histogram gradient boosting selected by five-fold, whole-trip cross-validation on 80 training trips. On the ten student-testing trips, mean trip-level Top-1 is **55.86%** for nearest road, **89.30%** for the selected classifier and **91.22%** after decoding. Pooling all 19,160 student-test observations gives **89.49%** after decoding. Mean normalised edit distance against the supplied ordered route improves from **0.407** to **0.172**. Lower edit distance is better.

Point labels were derived from the ordered reference route by monotone minimum-distance alignment. Direction and timing are ambiguous at stops, junctions and repeated visits. The scorer includes all points, including candidate misses and high-residual examples. The largest training trips can produce long stretches far from the reference. Two trips account for 17,052 of 18,978 training candidate misses. All ten student-test trips together have 63 misses. These are development results. The teacher's official point alignment, if supplied, should replace the private scoring function before evaluation.

[Development report](results/development_report.pdf) · [Per-model results](results/model_comparison.csv) · [Per-trip predictions evaluated by method](results/route_comparison.csv) · [Partition inventory](results/inventory.csv) · [Bayesian trials](results/bayesian_trials.csv)

## Reproduce

From the project root, use Python 3.14 and the versions in requirements.txt. Model loading uses scikit-learn's saved estimator format. Install dependencies once on the inference laptop.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2
python -m pathsnap.prepare --data 'Student folder'
python -m pathsnap.experiments
python -m pathsnap.route_eval --data 'Student folder'
python -m pathsnap.checks
python -m pathsnap.maps --folium
python -m pathsnap.report
```

The data root must contain train and student_test, each containing trip directories with one track, nodes, arcs and route file. Training features and derived references are local files under .cache/features. Rebuild that directory in a fresh location when changing the dataset. The saved classifier and decoder settings in artifacts/ suffice for prediction. Only load trusted model artifacts.

The experiment samples up to 256 covered observations per fitting trip, keeping all candidates for each sampled observation. Validation and student-test scoring include every observation. Model selection uses the mean of per-trip Top-1 over held-out training trips. The GP search performed six trials over the winning family's learning rate. Tuning changed CV accuracy by less than 0.001 percentage point and slightly reduced student-test accuracy. Folds, choices and timings are recorded under results/.

The training-only decoder fit takes scale, speed and gap estimates from training percentiles. Continuity in model_comparison.csv measures equal or immediately adjoining arcs between observation pairs. In route_comparison.csv it measures adjacency after removing repeated observations on the same arc and reconstructing connectors. A break is an explicit separator. The two continuity columns measure different things. Score timings in model_comparison.csv cover scoring prepared features. api_checks.csv measures the full worker through the FastAPI handler including loading, features, decoding and scoring.

## Two-laptop demo

On Laptop 1 start the service from the project root. The two data paths below enable scoring on permitted development trips. Set them to the evaluator's input partition and private scoring location only for the final evaluation. A separate private reference directory, when supplied, should contain a subdirectory per trip with its .route file. Omitting PATHSNAP_REFERENCES runs prediction only.

```bash
export PATHSNAP_TRIPS='Student folder/student_test'
export PATHSNAP_REFERENCES='Student folder/student_test'
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2
python -m uvicorn pathsnap.api:app --host 0.0.0.0 --port 8000
```

On Laptop 2 copy the ui directory, open a terminal in its parent directory and run:

```bash
python -m http.server 8080 --directory ui
```

Open http://localhost:8080 on Laptop 2, enter Laptop 1's LAN address and API port, load the trips, select one, and choose **Reconstruct route**. The UI draws each decoded fragment separately and works without map tiles. The API logs the trip, observation count, fragment count, aggregate agreement and processing time. A successful GET /trips returns the configured catalog. POST /predict/{trip_id} runs inference from disk on each request. The optional PATHSNAP_PYTHON setting selects a separate worker interpreter when the API and ML packages use different environments.

### Student-test rehearsal and evidence

1. Put both laptops on the same LAN. On Laptop 1, run the Uvicorn command above and obtain its LAN IP address from the active network settings. Use `http://<Laptop-1-LAN-IP>:8000` as the API address on Laptop 2.
2. On Laptop 2, start the UI with the command above. Open `http://localhost:8080`, enter the API address and click **Load trips**. Check that the selector contains all ten IDs listed in [the development API results](results/api_checks.csv). If the list fails to load, open the API address with `/trips` appended in Laptop 2's browser to check reachability. If needed, allow inbound TCP port 8000 on Laptop 1 and use a network that allows the laptops to reach each other. Allow local network access if the browser prompts for it.
3. Reconstruct `00000000`, then `00000007`. After each request, check that the displayed trip ID, GPS points, predicted route and aggregate score update. The committed student-test handler results give 2,503 observations and 96.20% Top-1 for `00000000`, and 1,354 observations and 89.07% for `00000007`.
4. In Laptop 2's browser developer tools, inspect each `POST /predict/<trip_id>` in the Network panel. Check for HTTP 200 and confirm the JSON contains `trip`, `track`, `route`, `observation_arcs`, `fragment_starts`, `metrics` and `elapsed_seconds`. The score fields are `metrics.observations` and `metrics.top1`. The two geometry fields must show GPS points and the prediction. Check that the response has no reference labels or reference geometry. Close the developer tools before taking the UI screenshot.
5. For one successful request, take screenshots on both laptops before issuing another prediction. Save the Laptop 2 browser view as `results/demo_ui.png` with the trip ID, GPS, route and result visible. Save the Laptop 1 Uvicorn terminal view as `results/demo_api_terminal.png` with the matching `POST prediction trip=...` result and the `POST /predict/...` HTTP 200 access log visible. Record the demonstrated IDs, commands used and any setup fix in the rehearsal record.

Local smoke test for [issue #2](https://github.com/AtharvMitt/ML-Hackathon/issues/2): Chrome and the real FastAPI worker ran on one Mac with a synthetic junction in place of the student-test data. Chrome loaded all ten fixture IDs and rendered two successive HTTP 200 predictions using the saved model. GPS points, route geometry and the aggregate observation count updated. The response exposed GPS, predictions, runtime and aggregate scoring fields. The two-laptop student-test rehearsal and its paired screenshots are pending.

For the standalone predictor, supply a trip directory with only .track, .nodes and .arcs:

```bash
python -m pathsnap.predict '/path/to/a/trip'
```

## Verification and final submission

The inference feature matrix contains distances, motion, direction and segment geometry. IDs only locate files or index geometry. The checks exercise reference removal, new trip names, node and arc permutations, and two trajectories taking different exits from the same synthetic junction. The FastAPI handlers were exercised for all ten development trips and reject unknown IDs.

The static trip overlay and development PDF can be regenerated offline with the ML dependencies using `python -m pathsnap.maps` followed by `python -m pathsnap.report`. Folium HTML generation additionally requires Folium. Render that interactive map, include a Folium export in the team's final report, capture the live two-laptop UI and receiving terminal, and verify the repository is accessible to the evaluators. Browser/LAN demonstration evidence is outstanding. The public project repository is [AtharvMitt/ML-Hackathon](https://github.com/AtharvMitt/ML-Hackathon). The PDF is a development draft produced with AI assistance. The team should independently review and be able to reproduce and explain its submission.
