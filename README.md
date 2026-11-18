# PathSnap

PathSnap reconstructs the directed roads a vehicle travelled from its GPS observations and the supplied road network. Laptop 1 serves a trained two-stage histogram gradient boosting model. Laptop 2 selects a trip and displays its reconstructed route.

## Where to find each part

| To inspect or run | Start here |
|---|---|
| Two-laptop evaluator demo | [Installation and commands for both laptops](#run-the-two-laptop-demo) |
| Five classifiers and Bayesian tuning | [Training notebook](Train_and_Evaluate_Models.ipynb), sections 6 and 7 · [Five-model comparison](results/model_comparison.csv) · [Bayesian trials](results/bayesian_trials.csv) |
| Final demo model | [Notebook section 9](Train_and_Evaluate_Models.ipynb#sequence-results) · [Training and evaluation guide](MODEL_TRAINING_AND_EVALUATION.md) · [Saved two-stage model](artifacts/temporal.joblib) |
| Raw trips and cleaned data | [Original training and student-test trips](Student%20folder/) · [Original evaluator trips](evaluator_test/) · [Cleaned data](cleaned_data/) · [Example training CSV](cleaned_data/train/00000001.csv) |
| Prepare evaluator inputs | [Preparation notebook](Prepare_Evaluator_Test_Data.ipynb) · [Executed copy](cleaned_data/evaluator_test/Prepare_Evaluator_Test_Data.executed.ipynb) |
| Verify final results | [Training-fold configuration comparison](notebook_run/temporal_cv.csv) · [Student-test and validation metrics by trip](notebook_run/sequence_comparison.csv) · [Evaluator correct and total counts](notebook_run/temporal_evaluator_counts.csv) |
| Inspect the maps and interface | [Interactive Folium HTML](results/development_map.html) (open locally in a browser) · [Folium screenshot](results/folium_map.png) · [Demo UI screenshot](results/sequence_ui.png) |

## Run the two-laptop demo

Connect both laptops to the same Wi-Fi. From the repository root on **Laptop 1**, install the dependencies once using Python 3.14 and start the evaluator API. The saved model and prepared evaluator inputs are included.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
PATHSNAP_SHOW_REFERENCE=0 ./run_local.sh --lan --evaluator
```

The terminal prints Laptop 1’s network addresses. Use its Wi-Fi IPv4 address on Laptop 2. The default API port is 8000. If occupied, set `API_PORT=8001` for the launcher and use that port in the browser. The command above keeps evaluator reference geometry on the server and returns the agreement score after prediction. For a rehearsal with the reference overlay, use `./run_local.sh --lan --evaluator`. For student-test trips, use `./run_local.sh --lan`.

On **Laptop 2**, from the repository root, start the UI:

```bash
python3 -m http.server 8080 --bind 127.0.0.1 --directory ui
```

Open `http://127.0.0.1:8080`. Enter `http://<Laptop-1-Wi-Fi-IP>:8000` in **API address**, click **Load trips**, select a trip, then click **Reconstruct route**. The grey roads come from that trip’s network. The UI can toggle GPS observations, the prediction, and the reference route when available. For a single-laptop rehearsal, run `./run_local.sh --evaluator`.

## Accuracy of the model on main

| Partition | Correct observations | Pooled Top-1 | Mean trip Top-1 |
|---|---:|---:|---:|
| Student test (10 trips) | 17,969 / 19,160 | 93.78% | 94.58% |
| Evaluator test (10 trips) | 18,904 / 19,674 | 96.09% | 96.27% |

A prediction is correct when its directed road agrees with the point label derived from the supplied reference traversal. Pooled Top-1 weights every observation equally. Mean trip Top-1 weights every trip equally.

## Reproduce the experiments

Install the notebook dependencies in the Laptop 1 environment and open the training notebook from the repository root.

```bash
python -m pip install pandas jupyterlab
jupyter lab Train_and_Evaluate_Models.ipynb
```

Sections 1–5 rebuild the candidate tables from the raw data. Sections 6–8 reproduce the five-model experiment, Bayesian tuning, and original Viterbi comparison. Section 9 uses the cleaned tables and recorded cross-validation results to select the sequence configuration, refit on 80 training trips, and write `artifacts/temporal.joblib`. Its scoring cell recomputes student-test results and displays recorded evaluator counts by default. To rescore an evaluator ZIP, set `PATHSNAP_EVALUATOR_ZIP` to its path before launching Jupyter.

The original tuned single-stage booster is stored in `artifacts/model.joblib`. The API preferentially loads `artifacts/temporal.joblib`. The [training guide](MODEL_TRAINING_AND_EVALUATION.md) explains both experiments and their respective scores.

To check predictions after a refit, run `python -m pathsnap.checks`. To reproduce the per-trip comparison of sequence variants, run `python -m pathsnap.sequence_comparison`. To regenerate the Folium map from the trip data, run `python -m pathsnap.maps --folium`.

The [development report](results/development_report.pdf) records the original experiment and was prepared with AI assistance.
