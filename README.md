# PathSnap

PathSnap matches a GPS track to the directed roads in its supplied network. Laptop 1 runs a candidate classifier, a Viterbi decoder and private scoring. A browser on Laptop 2 selects a trip and displays the reconstructed route.

## Current results

The saved model is histogram gradient boosting selected by five-fold, whole-trip cross-validation on 80 training trips. On the ten student-testing trips, mean trip-level Top-1 is **55.86%** for nearest road, **89.30%** for the selected classifier and **91.22%** after decoding. Pooling all 19,160 student-test observations gives **89.49%** after decoding. Mean normalised edit distance against the supplied ordered route improves from **0.407** to **0.172**. Lower edit distance is better.

Point labels were derived from the ordered reference route by monotone minimum-distance alignment. Direction and timing are ambiguous at stops, junctions and repeated visits. The scorer includes all points, including candidate misses and high-residual examples. The largest training trips can produce long stretches far from the reference. Two trips account for 17,052 of 18,978 training candidate misses. All ten student-test trips together have 63 misses. These are development results. The teacher's official point alignment, if supplied, should replace the private scoring function before evaluation.

[Development report](results/development_report.pdf) · [Per-model results](results/model_comparison.csv) · [Per-trip predictions evaluated by method](results/route_comparison.csv) · [Partition inventory](results/inventory.csv) · [Bayesian trials](results/bayesian_trials.csv)

Read the [preprocessing, five-model comparison and MLOps explanation](PATHSNAP_ML_EXPLAINED.md) for the implementation details.

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

Connect both laptops to the same Wi-Fi. On Laptop 1, from the repository root, start the model and API with the student-test data and private scoring enabled.

```bash
git pull --ff-only origin main
./run_local.sh --lan
```

The API listens on all IPv4 interfaces. Give Laptop 2 the Wi-Fi address of Laptop 1 and the API port (default 8000). If that port is occupied, use `API_PORT=8001 ./run_local.sh --lan` and share port 8001. For the final evaluation, set `PATHSNAP_TRIPS` to the reserved input partition and `PATHSNAP_REFERENCES` to the private reference root. With a custom trip root, scoring is enabled only when a reference root is supplied.

On Laptop 2, run these commands from the repository root. Only Python 3 and the UI files are needed on this laptop.

```bash
git pull --ff-only origin main
python3 -m http.server 8080 --bind 127.0.0.1 --directory ui
```

Open http://127.0.0.1:8080 on Laptop 2. Enter `http://<Laptop-1-Wi-Fi-IP>:8000` in **API address**, click **Load trips**, select a trip and click **Reconstruct route**. To test connectivity directly, open `http://<Laptop-1-Wi-Fi-IP>:8000/trips` on Laptop 2. For a one-laptop rehearsal, run `./run_local.sh` without arguments.

The UI draws the selected trip’s road network in grey, the decoded fragments in green and GPS observations in purple. The geometry comes from the API and works offline. The API logs the trip, observation count, fragment count, aggregate agreement and processing time. A successful GET /trips returns the configured catalog. POST /predict/{trip_id} runs inference from disk on each request. The optional PATHSNAP_PYTHON setting selects a separate worker interpreter when the API and ML packages use different environments.

For the standalone predictor, supply a trip directory with only .track, .nodes and .arcs:

```bash
python -m pathsnap.predict '/path/to/a/trip'
```

## Verification and final submission

The inference feature matrix contains distances, motion, direction and segment geometry. IDs only locate files or index geometry. The checks exercise reference removal, new trip names, node and arc permutations, and two trajectories taking different exits from the same synthetic junction. The FastAPI handlers were exercised for all ten development trips and reject unknown IDs.

The static trip overlay and development PDF can be regenerated offline with the ML dependencies using `python -m pathsnap.maps` followed by `python -m pathsnap.report`. Folium HTML generation additionally requires Folium. Render that interactive map, include a Folium export in the team's final report, capture the live two-laptop UI and receiving terminal, and verify the repository is accessible to the evaluators. Browser/LAN demonstration evidence is outstanding. The public project repository is [AtharvMitt/ML-Hackathon](https://github.com/AtharvMitt/ML-Hackathon). The PDF is a development draft produced with AI assistance. The team should independently review and be able to reproduce and explain its submission.
