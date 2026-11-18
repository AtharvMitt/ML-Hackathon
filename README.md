# PathSnap

PathSnap matches a GPS track to the directed roads in its supplied network. Laptop 1 runs a candidate classifier, a Viterbi decoder and private scoring. A browser on Laptop 2 selects a trip and displays the reconstructed route.

## Current results

The default predictor uses two histogram gradient boosting rankers. It combines GPS motion with directed road connectivity. Both stages were fitted on the 80 training trips. Whole-trip validation selected the final configuration before evaluator scoring.

| Split | Correct GPS observations | Pooled Top-1 | Mean trip Top-1 |
|---|---:|---:|---:|
| Student test (10 trips) | 17,969 / 19,160 | 93.78% | 94.58% |
| Evaluator (10 trips, local reference alignment) | 18,904 / 19,674 | 96.09% | 96.27% |

Section 9 of the [executed training notebook](PathSnap_Training.ipynb#sequence-results) reproduces the final fit from the cleaned tables. The demo loads `artifacts/temporal.joblib`. The evaluator reference is opened after prediction for scoring.

### Original five-model experiment

The original saved model is histogram gradient boosting selected by five-fold, whole-trip cross-validation on 80 training trips. On the ten student-testing trips, mean trip-level Top-1 is **55.86%** for nearest road, **89.30%** for the selected classifier and **91.22%** after decoding. Pooling all 19,160 student-test observations gives **89.49%** after decoding. Mean normalised edit distance against the supplied ordered route improves from **0.407** to **0.172**. Lower edit distance is better.

Point labels were derived from the ordered reference route by monotone minimum-distance alignment. Direction and timing are ambiguous at stops, junctions and repeated visits. The scorer includes all points, including candidate misses and high-residual examples. The largest training trips can produce long stretches far from the reference. Two trips account for 17,052 of 18,978 training candidate misses. All ten student-test trips together have 63 misses. These are development results. The teacher's official point alignment, if supplied, should replace the private scoring function before evaluation.

[Development report](results/development_report.pdf) · [Per-model results](results/model_comparison.csv) · [Per-trip predictions evaluated by method](results/route_comparison.csv) · [Partition inventory](results/inventory.csv) · [Bayesian trials](results/bayesian_trials.csv)

Read the [preprocessing, five-model comparison and MLOps explanation](PATHSNAP_ML_EXPLAINED.md) for the implementation details.

## Training notebook and cleaned data

[Open the executed training notebook](PathSnap_Training.ipynb) to inspect preprocessing, all five fitted models, Bayesian tuning and Viterbi decoding. It also runs independently beside the teacher's `Student folder/`. The [cleaned training and student-test CSVs](cleaned_data/) and [notebook experiment outputs](notebook_run/) accompany the notebook.

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

## Prepare the evaluator trips

Open [PathSnap_Evaluator_Preparation.ipynb](PathSnap_Evaluator_Preparation.ipynb) beside the teacher’s `evaluator_test/` folder and run all cells. The notebook writes validated track and network files, a per-trip candidate-feature CSV, and an inventory under `cleaned_data/evaluator_test/`. It uses NumPy, SciPy, pandas and Jupyter. The demo loads the saved model and decoder from `artifacts/` and performs fresh inference from the cleaned track and network files.

Use `./run_local.sh --evaluator` for a one-laptop check or `./run_local.sh --lan --evaluator` on Laptop 1 for the two-laptop demo. Laptop 2 uses the usual UI commands below. The launcher selects only the evaluator trips and defaults its scoring reference directory to the original `evaluator_test/`. Set `PATHSNAP_SHOW_REFERENCE=0` to keep the reference geometry server-side during official evaluation. Restart the server and click **Load trips** when switching partitions.

## Two-laptop demo

Connect both laptops to the same Wi-Fi. On Laptop 1, from the repository root, start the API for the prepared evaluator trips.

```bash
git pull --ff-only origin main
./run_local.sh --lan --evaluator
```

The API listens on all IPv4 interfaces. Give Laptop 2 the Wi-Fi address of Laptop 1 and the API port (default 8000). If that port is occupied, use `API_PORT=8001 ./run_local.sh --lan --evaluator` and share port 8001. Set `PATHSNAP_TRIPS` to the input partition and `PATHSNAP_REFERENCES` to the corresponding reference root. By default, a scored response includes the reference geometry for the UI toggle. The handout requires evaluator ground truth to stay server-side. To meet that requirement while still scoring, start the API with `PATHSNAP_SHOW_REFERENCE=0`. With a custom trip root, scoring is enabled only when a reference root is supplied.

On Laptop 2, run these commands from the repository root. Only Python 3 and the UI files are needed on this laptop.

```bash
git pull --ff-only origin main
python3 -m http.server 8080 --bind 127.0.0.1 --directory ui
```

Open http://127.0.0.1:8080 on Laptop 2. Enter `http://<Laptop-1-Wi-Fi-IP>:8000` in **API address**, click **Load trips**, select a trip and click **Reconstruct route**. To test connectivity directly, open `http://<Laptop-1-Wi-Fi-IP>:8000/trips` on Laptop 2. For a one-laptop rehearsal, run `./run_local.sh` without arguments.

Use the independent **Predicted route** (solid green) and **Reference route** (dashed orange) checkboxes to compare traversals. Toggle **GPS observations** to show or hide the purple points over the grey road network. A missing or withheld reference disables its checkbox. The geometry comes from the API and works offline. The API logs the trip, observation count, fragment count, aggregate agreement and processing time. A successful GET /trips returns the configured catalog. POST /predict/{trip_id} runs inference from disk on each request. The optional PATHSNAP_PYTHON setting selects a separate worker interpreter when the API and ML packages use different environments.

For the standalone predictor, supply a trip directory with only .track, .nodes and .arcs:

```bash
python -m pathsnap.predict '/path/to/a/trip'
```

## Verification and final submission

The inference feature matrix contains distances, motion, direction and segment geometry. IDs only locate files or index geometry. The checks exercise reference removal, new trip names, node and arc permutations, and two trajectories taking different exits from the same synthetic junction. The FastAPI handlers were exercised for all ten development trips and reject unknown IDs.

The static trip overlay and development PDF can be regenerated offline with the ML dependencies using `python -m pathsnap.maps` followed by `python -m pathsnap.report`. Folium HTML generation additionally requires Folium. Render that interactive map, include a Folium export in the team's final report, capture the live two-laptop UI and receiving terminal, and verify the repository is accessible to the evaluators. Browser/LAN demonstration evidence is outstanding. The public project repository is [AtharvMitt/ML-Hackathon](https://github.com/AtharvMitt/ML-Hackathon). The PDF is a development draft produced with AI assistance. The team should independently review and be able to reproduce and explain its submission.
