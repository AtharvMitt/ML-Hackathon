# PathSnap model training and evaluation

The demo on `main` serves **two histogram gradient boosting classifiers** from `artifacts/temporal.joblib`. They rank candidate roads using the GPS track, road geometry, neighbouring observations, and directed connectivity. The saved model achieved **94.58% mean trip Top-1 on student test** and **96.27% on evaluator test**. Both figures measure agreement with point labels derived from the supplied reference route. [Final notebook section](Train_and_Evaluate_Models.ipynb#sequence-results) · [Evaluator counts](notebook_run/temporal_evaluator_counts.csv)

| Training notebook section | What to look for |
|---|---|
| 1–5 | Raw trip validation, reference alignment, candidate generation, the 11 base features, and readable cleaned CSVs. |
| 6 | The `models` dictionary defines five classifiers. The loop calling `compare` fits each on five whole-trip folds. |
| 7–8 | Six Bayesian learning-rate trials on the winning booster, its refit, and the original classifier versus Viterbi evaluation. |
| 9 | Selection from the saved sequence-validation counts, a fresh two-stage fit on 80 training trips, and scoring on the test partitions. |

[Training notebook](Train_and_Evaluate_Models.ipynb) · [Installation and demo commands](README.md#run-the-two-laptop-demo)

## Saved trained models

| Notebook cell | What it saves | Tracked file |
|---|---|---|
| Section 9, **Refit on the 80 training trips** | Both fitted gradient boosters, the candidate feature schema, and the selected sequence settings in one bundle. The demo loads this file directly. | [artifacts/temporal.joblib](artifacts/temporal.joblib) |
| Section 7, **Tune the selected family, refit and save** | The fitted single-stage booster selected after Bayesian tuning, with its feature order and candidate count. | [notebook_run/model.joblib](notebook_run/model.joblib) |

The packaged fallback is a copy of the Section 7 model. The route decoder reads its numerical settings from a separate JSON file. [Fallback classifier](artifacts/model.joblib) · [Decoder configuration](artifacts/decoder.json) · [Model-loading code](pathsnap/predict.py)

## 1. How raw files become labelled training examples

### Read and validate each trip

| File | What one row represents | How PathSnap uses it |
|---|---|---|
| `.track` | Longitude, latitude and timestamp in seconds | Provides observed locations, movement direction, displacement and timing. |
| `.nodes` | Longitude and latitude of a road-network node | Its zero-based row position identifies a node. |
| `.arcs` | Start-node ID and end-node ID | Its zero-based row position identifies a directed road segment. A road can have an arc for each direction. |
| `.route` | Arc ID at one position along the supplied reference traversal | Provides supervision during development and a reference for private scoring after prediction. |

The reader checks column counts, finite coordinates and timestamps, valid node indices, and nondecreasing timestamps. The reference reader checks that route indices exist in the trip’s arc table. The preparation script checks that a trip ID appears only once across the partitions. It also records gaps, network components, isolated nodes, zero-length segments and breaks in reference connectivity. Zero-length segments are excluded from the candidate search because they have no direction. Original row IDs are preserved so references still resolve correctly. [Trip reader](pathsnap/core.py) · [Preparation script](pathsnap/prepare.py)

| Recorded quantity | Training | Student testing |
|---|---:|---:|
| Trips | 80 | 10 |
| GPS observations | 208,417 | 19,160 |
| Nonfinite track or node values | 0 | 0 |
| Repeated timestamps within a trip | 0 | 0 |
| Zero-length arcs across trip networks | 100 | 6 |
| Largest interval between consecutive observations | 153,064 s | 210 s |

The pipeline retains the original sampling intervals. A long interval therefore remains visible to the decoder, which can start a new route fragment. Interpolating a continuous track through the largest training gap would imply knowledge of more than 42 hours of unobserved movement. [Dataset inventory](results/inventory.csv)

### Convert angles to metres and find candidate roads

The code centres the coordinate system at the mean longitude and latitude of the trip’s road-network nodes. It converts latitude differences to metres using Earth’s radius and scales longitude differences by the cosine of the centre latitude. This local approximation puts GPS displacements and road lengths in the same units. [Coordinate conversion](pathsnap/core.py)

For each GPS point, PathSnap projects the point onto nearby directed segments. The projection is clamped to each segment’s endpoints, so a point beyond the end of a road is measured against the endpoint. A spatial index accelerates the search by representing long arcs with shorter pieces of at most 40 metres. Exact point-to-segment distances then determine the nearest **eight** valid directed arcs, or all available arcs when fewer exist. The 40-metre value controls indexing resolution. It does not limit how far a point may lie from its candidates. [Candidate search](pathsnap/core.py)

### Derive one reference arc per observation

A GPS row records a **time sample**. A route row records an **arc traversal**. A vehicle can remain on one arc for dozens of samples or cross a short arc between samples. Joining the files by row number would assign incorrect labels.

For each GPS observation, the training labeler calculates the distance to each occurrence along the supplied reference route. Dynamic programming selects a sequence of route positions that minimises total squared distance while permitting the position to stay still or advance. A backtracking pass recovers an arc ID for each observation. Working with route *positions* preserves the distinction between an early and a later visit to the same arc. The result is a reproducible alignment, although the exact arc at a junction or stop can remain ambiguous. The deployed predictor works from the track and network alone. [Reference alignment](pathsnap/reference.py) · [Inference entry point](pathsnap/predict.py)

**Follow one training trip through cleaning.** Trip `00000001` contains 1,315 GPS samples and 525 reference traversal steps. Candidate generation produces 10,520 CSV rows, eight for each observation. Filter the CSV by `observation` to see the same GPS sample alongside eight competing arcs. The aligned reference occurs among those eight for 1,313 samples. For the remaining two, all eight rows have `is_match = 0`. Keeping those rows in the export makes candidate-search failures visible. [Trip CSV](cleaned_data/train/00000001.csv) · [Inventory](cleaned_data/inventory.csv)

### Why these five feature groups produce 11 inputs

The feature groups follow from what the teacher’s files let us measure about a vehicle and a directed road. At a junction, several arcs can be equally close. The direction of movement can distinguish opposite directions of travel on the same geometry, and the projected position can show whether the next movement would carry the vehicle past a segment’s end. Speed gives the tree models context for interpreting directional evidence when the vehicle barely moves. The experiments compare models using all five groups together. Measuring the gain from any one group would require retraining with that group removed.

| Evidence | Features | What the model learns from them |
|---|---|---|
| Proximity | `log_distance`, `distance_above_best` | How close the road is in absolute terms and how much another road improves the fit. |
| Segment geometry | `along_fraction`, `log_length` | Whether the projected point lies near an endpoint or in the interior of a long road. |
| Direction | `incoming_alignment`, `outgoing_alignment`, `window_alignment` | Cosine similarity between GPS movement and the directed arc. The wider window spans up to three samples on either side. |
| Speed | `incoming_speed`, `outgoing_speed` | How quickly the observed position changes on either side, using actual elapsed time. |
| Endpoint overshoot | `incoming_overshoot`, `outgoing_overshoot` | How far an adjacent GPS displacement exceeds the available length before or after the projection along that arc. |

The code applies `log(1 + value)` to distance, length, speed and overshoot quantities to compress large values. The direction features retain their sign. A stationary displacement gives zero alignment. Speed uses metres divided by elapsed seconds and takes the value zero when the interval is zero. Neighbour indices stop at the ends of the track. Because outgoing motion uses later observations, this feature extractor assumes the trip is available as a batch, as it is during the demo. Trip and arc IDs are kept for grouping, joins and reconstruction. The five initial classifiers receive the 11 numeric features. Section 9 extends these features with sequence context for the final model. [Feature definitions and formulas](pathsnap/core.py)

**A real example.** At the second observation of student-test trip `00000000`, arcs `17634` and `17630` were both approximately **0.368 metres** from the GPS point. Their incoming-direction similarities were **−0.970** and **+0.970**, respectively. The saved model assigned arc `17630` about **99.90%** of the normalised candidate score, and the derived reference also selected that arc. Distance alone could not resolve this case. The score describes the model’s preference among the candidates, rather than an independently verified probability of the vehicle’s true location. [Feature implementation](pathsnap/core.py) · [Saved model](artifacts/model.joblib)

### Which fields stay out of the classifier, and why

| Field or source | How the pipeline uses it | Reason for exclusion from the 11 inputs |
|---|---|---|
| Trip ID, observation index, candidate arc ID, start and end node IDs | Group trips, join rows and recover the road geometry | These identify records. An arc numbered 100 in another trip can describe an entirely different road. Connectivity enters through the decoder. |
| Raw longitude, latitude and timestamp | Preserve the teacher’s observations in the CSV and derive distances, directions and speeds | Relative geometry and elapsed time carry the intended evidence to new locations and times. |
| Candidate rank | Order the nearby arcs for inspection | Rank alone hides whether first and second place differ by centimetres or metres. The feature `distance_above_best` retains that margin. |
| Reference route, aligned arc ID, reference residual and `is_match` | Supply supervision, diagnose alignment quality and score completed predictions | Each depends on knowing the reference. Using any as a predictor would leak the answer. `is_match` is the binary training target. Training residuals also set the decoder’s spatial scale. |
| Road names, speed limits and lane counts | The supplied network specifies node coordinates and directed endpoints | Those attributes are unavailable in the inputs. All five models can run using the supplied files alone. |

### Form the supervised task and keep trips together

A candidate receives target **1** when its arc ID matches the aligned reference for that observation and **0** otherwise. An observation with eight candidates therefore supplies eight training rows. This formulation lets the same classifier score geometry on a new trip whose local arc IDs mean something entirely different.

The reference arc was absent from the candidate set for **18,978 of 208,417 training observations** and **63 of 19,160 student-test observations**. A classifier restricted to those candidates cannot recover the correct arc at that point. Fitting samples up to **256 covered observations per training trip**, with all their candidate rows. With seed 41, the original booster fit used 20,480 observations across the 80 training trips, giving 163,840 candidate rows. Each sampled observation contributes one matching arc and seven alternatives. The cap limits the influence of extremely long trips. Validation and student-test Top-1 include every observation, including misses. Reported log loss uses covered observations. [Training-row construction and metrics](pathsnap/experiments.py) · [Coverage by trip](results/inventory.csv)

## 2. The five models and why boosting won

The teacher’s *Models and Training Guidelines*, sections 2–4, require at least five trained models and leave algorithm selection to the team. We chose five ways of scoring the same numeric candidate table. Each produces a binary match score that can feed the same route decoder. This lets us compare how the classifiers weigh evidence while holding the candidate search, feature columns and evaluation metric fixed.

All five classifiers learned the same binary candidate task. **Five-fold grouped cross-validation** assigned whole trips to folds. This keeps neighbouring points from one journey together and avoids an artificially easy validation task. Every model used the same folds and candidate features. Logistic regression’s scaler was fitted within each training fold. The selection objective was the average held-out **Top-1 accuracy per trip**, giving a short trip and a long trip equal weight. [Fold assignments](results/folds.csv) · [Experiment code](pathsnap/experiments.py)

| Model | Why we included it and how it scores a candidate | CV Top-1 | Student-test Top-1 | Student-test macro F1 |
|---|---|---:|---:|---:|
| Logistic regression | Tests whether one additive rule is sufficient. It learns a weight for each standardised feature, with up to 300 optimisation iterations. | 83.71% | 85.43% | 0.689 |
| Gaussian naive Bayes | Tests a simple probabilistic account of a match. It combines class-specific feature distributions under a conditional-independence assumption, which related distances can violate. | 83.57% | 85.46% | 0.705 |
| Decision tree | Tests whether interactions help, such as direction mattering more when two roads are equally close. Depth 12 and at least 32 rows per leaf limit how specific its rules can get. | 86.19% | 88.71% | 0.791 |
| Extra Trees | Tests whether averaging 32 randomised trees gives more stable predictions across trips than relying on one set of splits. Maximum depth 18 and at least eight rows per leaf. | 86.36% | 88.47% | 0.775 |
| **Histogram gradient boosting** | Tests whether successive trees can correct earlier scoring errors on ambiguous candidates. Histogram bins keep numerical splits efficient. It uses 100 iterations with early stopping disabled. | **86.98%** | **89.41%** | **0.805** |

These are the **initial comparison** results. Top-1 asks whether the highest-scoring arc matched the point reference. Macro F1 averages the F1 scores of the road-arc classes within a trip, then averages the trips. The separate nearest-road baseline achieved **55.86%** student-test Top-1. Its role was to measure what geometric proximity alone could accomplish. [Full comparison, including precision, recall, log loss and timing](results/model_comparison.csv)

### What Bayesian optimisation changed

Boosting had the best grouped CV Top-1, so the search tuned its learning rate. It considered 33 logarithmically spaced values from **0.025 to 0.3** and evaluated **six**. After three initial trials, a Gaussian-process model used the measured scores and their uncertainty to choose subsequent values by expected improvement. Each trial reused the same five grouped folds.

The selected learning rate was **0.0686053**. CV Top-1 moved from **86.97562% to 86.97568%**, effectively a tie. Student-test Top-1 changed from **89.41% to 89.30%**, while covered-point log loss improved slightly from **0.3343 to 0.3330**. This tuned single-stage booster is saved as `artifacts/model.joblib`. The later two-stage demo uses a fixed learning rate of 0.075 in both stages. Its configuration was selected separately using whole-trip validation. The experiment supports choosing the boosting family, but provides little evidence that learning-rate tuning improved accuracy. [Trial history](results/bayesian_trials.csv) · [Selected configuration](results/config.json)

The table uses the original run’s mean per-trip classifier timings on prepared features. Log loss includes observations whose reference arc is among the candidates. [Recorded comparison](results/model_comparison.csv)

| Single-stage booster | CV macro F1 | Student-test Top-1 | Student-test macro F1 | Covered log loss | Mean scoring time per trip |
|---|---:|---:|---:|---:|---:|
| Initial | 0.7889 | 89.41% | 0.8048 | 0.3343 | 35.5 ms |
| After Bayesian tuning | 0.7900 | 89.30% | 0.8016 | 0.3330 | 37.5 ms |

### Did the model train on the test data?

Classifier fitting, fold assignment, model selection, learning-rate selection and decoder parameter estimation all use the 80 training trips. The ten `student_test` trips supply development comparisons. The separate evaluator preparation notebook validates and exports the teacher’s new trips under `cleaned_data/evaluator_test/` for inference. It performs no model fitting. The demo computes its prediction before loading an available reference for scoring or display. [Training workflow](pathsnap/experiments.py) · [Evaluator preparation notebook](Prepare_Evaluator_Test_Data.ipynb) · [Demo worker](pathsnap/serve_worker.py)

For the original tuned single-stage classifier, mean trip-level Top-1 was **87.67% on the training trips** and **86.98% on held-out training folds**, a gap of **0.69 percentage points**. That small gap gives little indication of severe overfitting on these trips. The same folds guided model selection and tuning, so their score can be optimistic. The student-test figures describe the supplied development trips. Performance on new trips must be measured separately. [Recorded scores](results/model_comparison.csv)

## 3. Original classifier and HMM/Viterbi experiment

The classifier predicts a binary match score for each candidate. PathSnap clips the scores for numerical stability, converts them to log odds and applies a softmax within each observation. The resulting candidate distribution supplies the HMM emission scores. At each time step, the hidden state is a candidate directed arc together with the GPS point’s projected position on it. [Candidate scoring](pathsnap/experiments.py) · [Decoder](pathsnap/decoder.py)

The decoder compares movement along the directed network with the displacement observed by GPS. A transition can use the remaining portion of the previous arc, a directed shortest path between endpoints, and the initial portion of the next arc. For each predecessor, the decoder gives feasible transitions a log weight of `−|network distance − GPS displacement| / spatial scale` and normalises the weights over its feasible successors. A short backwards shift in projection on the same arc is allowed within the spatial scale to accommodate GPS jitter. Viterbi keeps the best cumulative log score and predecessor for each candidate, then backtracks to recover the sequence and its intermediate connector arcs.

Three estimates come from the training trips. The 75th percentile of distances to aligned reference arcs gives a **10.92 m** spatial scale. The 99th percentile of observed GPS speeds gives **46.35 m/s** for the transition search bound. The 99.5th percentile of positive sampling intervals gives a **6 s** gap threshold. A nonpositive interval, a gap exceeding that threshold or a step with no feasible continuation starts a new fragment. The HMM is the sequence stage applied to the chosen classifier. Its Viterbi search considers candidate roads in the supplied graph. The earlier alignment for generating training labels searches only positions along the known reference traversal. These settings and the chosen classifier are reused across trips. [Decoder parameter estimation](pathsnap/route_eval.py) · [Deployed values](artifacts/decoder.json)

| Student-test metric | Tuned classifier alone | Classifier + Viterbi |
|---|---:|---:|
| Mean trip-level Top-1 | 89.30% | **91.22%** |
| Mean trip-level arc macro F1 | 0.802 | **0.904** |
| Mean normalised route edit distance, lower is better | 0.407 | **0.172** |

Pooling all 19,160 observations gives **89.49%** decoded Top-1. The difference from 91.22% comes from weighting each observation equally instead of each trip equally. The sequence metric also answers a different question. On trip `00000007`, point-level agreement fell from **95.13% to 89.07%**, while normalised route edit distance improved from **0.401 to 0.024**. The decoded traversal closely followed the supplied arc sequence even though more individual observations disagreed with the derived point alignment. [Per-trip route results](results/route_comparison.csv)

## 4. Training the two-stage model used by the demo

1. **Prepare the inputs.** Section 9 reads the 80 per-trip CSVs in `cleaned_data/train/`. Each observation has up to eight candidate directed arcs. For each candidate, the first stage expands the base features to 68 inputs using motion over time and distance, repeated evidence for the same arc, and plausible travel through the directed network. The binary target is whether the candidate matches the aligned reference. [Feature construction](pathsnap/temporal.py) · [Training data reader](pathsnap/temporal_training.py)
2. **Select a configuration.** The notebook reads 28 recorded configurations from `temporal_cv.csv`. Each has one prediction-count row for each of the 80 training trips, across five folds. It ranks configurations by mean trip Top-1. The selected `stack` variant scored **91.99%** (unrounded 0.9199498), with path weight **0.25** and alignment strength **0.5**. Section 9 reads these previously generated fold results and then reruns the final fit. This selection is a comparison of recorded configurations. The Bayesian search in Section 7 tuned the original single-stage booster. [Configuration scores](notebook_run/temporal_cv.csv) · [Sequence comparison](notebook_run/sequence_comparison.csv)
3. **Train both stages.** Within the final fit, seed 41 shuffles training trip IDs into three groups. For each group, a first-stage booster trained on the other two groups scores its trips. Those out-of-fold scores supply 16 more features for the second stage (84 total). Stage 1 uses up to 768 covered observations per trip and 120 boosting iterations. Stage 2 uses up to 1,536 and 200 iterations. Both use learning rate 0.075, up to 31 leaves, at least 40 candidate rows per leaf, and disabled early stopping. After fitting stage 2, the notebook refits stage 1 across all 80 trips and saves both estimators together. [Final fit implementation](pathsnap/temporal_training.py)
4. **Predict and score.** The second-stage scores, forward and backward path costs, and ordered alignment select one directed arc per GPS observation. The graph decoder receives those selections and reconstructs the connecting route fragments. The API opens the available reference after prediction for scoring and optional display. Consequently the original 89.30% to 91.22% Viterbi comparison in Section 3 measures the earlier experiment. [Inference](pathsnap/predict.py) · [API scoring](pathsnap/serve_worker.py)

| Final-model evaluation | Student test | Evaluator test |
|---|---:|---:|
| Correct / total GPS observations | 17,969 / 19,160 | 18,904 / 19,674 |
| Pooled Top-1 | 93.78% | 96.09% |
| Mean trip Top-1 | 94.58% | 96.27% |

Both partitions contain ten trips. Section 9 scores the student-test tables directly. It displays saved evaluator counts by default and can recompute them with the evaluator ZIP. [Executed evaluation](Train_and_Evaluate_Models.ipynb#sequence-results) · [Evaluator results by trip](notebook_run/temporal_evaluator_counts.csv)

## 5. MLOps in this project

Here, **MLOps** is the engineering that lets the team reproduce an experiment and serve the selected model through an API. The training notebook provides the submission workflow. The Python modules implement the preparation, experiments, packaging and serving used by the demo. Experiment tracking uses CSV and JSON files in Git, deployment uses FastAPI and a Python worker, and operational visibility comes from request logs.

| Stage | Implementation | What can be checked afterward |
|---|---|---|
| Prepare data | `pathsnap.prepare` reads the two permitted partitions and writes one local `.npz` cache per trip with features, candidate IDs, aligned labels and edges. | `results/inventory.csv` records data quality and candidate misses. The cache can be rebuilt from the teacher’s data. |
| Run experiments | `pathsnap.experiments` uses shared features, seed 41 and recorded grouped folds. It fits all five families, evaluates them and searches the winner. | Fold membership, trial settings, per-trip metrics, aggregate metrics and timings are saved. |
| Package the predictor | `artifacts/temporal.joblib` stores both deployed rankers and their sequence settings. `artifacts/model.joblib` stores the original tuned booster. `artifacts/decoder.json` stores the learned decoder settings. | Inference checks the feature names before scoring. Shared preprocessing code keeps training and prediction feature calculations consistent. |
| Serve a request | Laptop 2 fetches `GET /trips` and sends `POST /predict/{trip_id}`. Laptop 1 discovers the trip directory, launches a fresh worker, loads the artifacts and reconstructs the route. | The server logs the trip, observation count, fragment count, aggregate metrics and elapsed time. The UI draws the returned road network, GPS points and predicted route. |
| Score and verify | When a private reference directory is configured, the worker aligns the reference and calculates accuracy **after** decoding. Regression checks exercise renamed trips, removed references, renumbered roads and changed synthetic turns. | Saved server checks cover the ten student-test trips. The UI can toggle predicted route, reference route when available, and GPS observations independently. Setting `PATHSNAP_SHOW_REFERENCE=0` suppresses the reference line in the response while retaining post-prediction scoring. |

Two details matter when explaining the system. First, the saved classifier timings cover scoring **already prepared features**. The API measurement includes worker startup, input loading, feature generation, decoding and optional scoring. Second, changing the data or model requires manually rebuilding the corresponding artifacts. The repository’s experiment records and request logs support reproducibility and diagnosis. Continuous retraining, a hosted model registry and drift alerts are outside the implemented workflow. [API](pathsnap/api.py) · [Inference worker](pathsnap/serve_worker.py) · [Prediction checks](pathsnap/checks.py) · [API timings](results/api_checks.csv)
