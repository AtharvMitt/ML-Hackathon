# PathSnap — preprocessing, five models, and MLOps

PathSnap reconstructs a vehicle’s route from a noisy GPS track and a directed road network. It makes two decisions. First, a classifier scores each nearby road segment for each GPS observation. Then a Viterbi decoder uses the road network and the order of the observations to choose a consistent sequence. **Histogram gradient boosting won the five-model comparison. The deployed classifier scored 89.30% mean trip-level Top-1 accuracy on the ten student-test trips. Decoding raised that to 91.22%.** These figures measure agreement with point labels derived from the supplied reference routes. [Model comparisons](results/model_comparison.csv) · [Route comparisons](results/route_comparison.csv)

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

### Build 11 features for each observation–candidate pair

| Evidence | Features | What the model learns from them |
|---|---|---|
| Proximity | Distance to the candidate; excess distance above the nearest candidate | How close the road is in absolute terms and how much another road improves the fit. |
| Segment geometry | Fraction along the segment; segment length | Whether the projected point lies near an endpoint or in the interior of a long road. |
| Direction | Incoming, outgoing and wider-window alignment | Cosine similarity between GPS movement and the directed arc. The wider window spans up to three samples on either side. |
| Speed | Incoming and outgoing speed | How quickly the observed position changes on either side, using actual elapsed time. |
| Endpoint overshoot | Incoming and outgoing overshoot | How much an adjacent GPS displacement exceeds the available length before or after the projection along that arc. |

The code applies `log(1 + value)` to distance, length, speed and overshoot quantities to compress large values. The direction features retain their sign. Because outgoing motion uses later observations, this feature extractor assumes the trip is available as a batch, as it is during the demo. Trip and arc IDs are kept for grouping, joins and reconstruction. The classifier receives the 11 numeric features. [Feature definitions and formulas](pathsnap/core.py)

**A real example.** At the second observation of student-test trip `00000000`, arcs `17634` and `17630` were both approximately **0.368 metres** from the GPS point. Their incoming-direction similarities were **−0.970** and **+0.970**, respectively. The saved model assigned arc `17630` about **99.90%** of the normalised candidate score, and the derived reference also selected that arc. Distance alone could not resolve this case. The score describes the model’s preference among the candidates, rather than an independently verified probability of the vehicle’s true location. [Feature implementation](pathsnap/core.py) · [Saved model](artifacts/model.joblib)

### Form the supervised task and keep trips together

A candidate receives target **1** when its arc ID matches the aligned reference for that observation and **0** otherwise. An observation with eight candidates therefore supplies eight training rows. This formulation lets the same classifier score geometry on a new trip whose local arc IDs mean something entirely different.

The reference arc was absent from the candidate set for **18,978 of 208,417 training observations** and **63 of 19,160 student-test observations**. A classifier restricted to those candidates cannot recover the correct arc at that point. Fitting samples up to **256 covered observations per training trip**, with all their candidate rows. The final fit used 20,480 observations across the 80 trips. The cap limits the influence of extremely long trips. Validation and student-test Top-1 include every observation, including misses. Reported log loss uses covered observations. [Training-row construction and metrics](pathsnap/experiments.py) · [Coverage by trip](results/inventory.csv)

## 2. The five models and why boosting won

All five classifiers learned the same binary candidate task. **Five-fold grouped cross-validation** assigned whole trips to folds. This keeps neighbouring points from one journey together and avoids an artificially easy validation task. Every model used the same folds and candidate features. Logistic regression’s scaler was fitted within each training fold. The selection objective was the average held-out **Top-1 accuracy per trip**, giving a short trip and a long trip equal weight. [Fold assignments](results/folds.csv) · [Experiment code](pathsnap/experiments.py)

| Model | How it scores a candidate | CV Top-1 | Student-test Top-1 | Student-test macro F1 |
|---|---|---:|---:|---:|
| Logistic regression | Learns one weighted combination of standardised features. Up to 300 optimisation iterations. | 83.71% | 85.43% | 0.689 |
| Gaussian naive Bayes | Estimates feature distributions for matching and nonmatching candidates. Its conditional-independence assumption is imperfect for related features such as distance and excess distance. | 83.57% | 85.46% | 0.705 |
| Decision tree | Learns conditional splits, allowing the effect of direction to depend on geometry. Maximum depth 12 and at least 32 training rows per leaf. | 86.19% | 88.71% | 0.791 |
| Extra Trees | Averages 32 randomised trees. Maximum depth 18 and at least eight rows per leaf. | 86.36% | 88.47% | 0.775 |
| **Histogram gradient boosting** | Bins feature values and adds trees sequentially to reduce binary log loss. Configured for 100 boosting iterations with early stopping disabled. | **86.98%** | **89.41%** | **0.805** |

These are the **initial comparison** results. Top-1 asks whether the highest-scoring arc matched the point reference. Macro F1 averages the F1 scores of the road-arc classes within a trip, then averages the trips. The separate nearest-road baseline achieved **55.86%** student-test Top-1. Its role was to measure what geometric proximity alone could accomplish. [Full comparison, including precision, recall, log loss and timing](results/model_comparison.csv)

### What Bayesian optimisation changed

Boosting had the best grouped CV Top-1, so the search tuned its learning rate. It considered 33 logarithmically spaced values from **0.025 to 0.3** and evaluated **six**. After three initial trials, a Gaussian-process model used the measured scores and their uncertainty to choose subsequent values by expected improvement. Each trial reused the same five grouped folds.

The selected learning rate was **0.0686053**. CV Top-1 moved from **86.97562% to 86.97568%**, effectively a tie. Student-test Top-1 changed from **89.41% to 89.30%**, while covered-point log loss improved slightly from **0.3343 to 0.3330**. The deployed classifier follows the CV selection rule. The experiment supports choosing the boosting family, but provides little evidence that learning-rate tuning improved accuracy. [Trial history](results/bayesian_trials.csv) · [Selected configuration](results/config.json)

## 3. How independent scores become a route

The classifier predicts a binary match score for each candidate. PathSnap clips the scores for numerical stability, converts them to log odds and applies a softmax within each observation. The resulting candidate distribution supplies the HMM emission scores. At time step (t), the hidden state is a candidate directed arc together with the GPS point’s projected position on it. [Candidate scoring](pathsnap/experiments.py) · [Decoder](pathsnap/decoder.py)

The decoder compares movement along the directed network with the displacement observed by GPS. A transition can use the remaining portion of the previous arc, a directed shortest path between endpoints, and the initial portion of the next arc. The transition score favours plausible travel distances. Viterbi keeps the best cumulative log score and predecessor for each candidate, then backtracks to recover the sequence and its intermediate connector arcs.

Training percentiles supplied a **10.92 m** spatial residual scale, a **46.35 m/s** speed estimate used in the search bound and a **6 s** gap threshold. A gap exceeding the threshold, or a step with no feasible continuation, starts a new fragment. These settings and the chosen classifier are reused across trips. [Decoder parameter estimation](pathsnap/route_eval.py) · [Deployed values](artifacts/decoder.json)

| Student-test metric | Tuned classifier alone | Classifier + Viterbi |
|---|---:|---:|
| Mean trip-level Top-1 | 89.30% | **91.22%** |
| Mean trip-level arc macro F1 | 0.802 | **0.904** |
| Mean normalised route edit distance, lower is better | 0.407 | **0.172** |

Pooling all 19,160 observations gives **89.49%** decoded Top-1. The difference from 91.22% comes from weighting each observation equally instead of each trip equally. The sequence metric also answers a different question. On trip `00000007`, point-level agreement fell from **95.13% to 89.07%**, while normalised route edit distance improved from **0.401 to 0.024**. The decoded traversal closely followed the supplied arc sequence even though more individual observations disagreed with the derived point alignment. [Per-trip route results](results/route_comparison.csv)

## 4. MLOps in this project

Here, **MLOps** is the engineering that lets the team reproduce an experiment and serve the selected model through an API. The repository implements a manually run, script-based workflow. Experiment tracking uses CSV and JSON files in Git, deployment uses FastAPI and a Python worker, and operational visibility comes from request logs.

| Stage | Implementation | What can be checked afterward |
|---|---|---|
| Prepare data | `pathsnap.prepare` reads the two permitted partitions and writes one local `.npz` cache per trip with features, candidate IDs, aligned labels and edges. | `results/inventory.csv` records data quality and candidate misses. The cache can be rebuilt from the teacher’s data. |
| Run experiments | `pathsnap.experiments` uses shared features, seed 41 and recorded grouped folds. It fits all five families, evaluates them and searches the winner. | Fold membership, trial settings, per-trip metrics, aggregate metrics and timings are saved. |
| Package the predictor | `artifacts/model.joblib` stores the selected estimator, ordered feature names and candidate count. `artifacts/decoder.json` stores the learned decoder settings. | Inference checks the feature names before scoring. Shared preprocessing code keeps training and prediction feature calculations consistent. |
| Serve a request | Laptop 2 fetches `GET /trips` and sends `POST /predict/{trip_id}`. Laptop 1 discovers the trip directory, launches a fresh worker, loads the artifacts and reconstructs the route. | The server logs the trip, observation count, fragment count, aggregate metrics and elapsed time. The UI draws the returned road network, GPS points and predicted route. |
| Score and verify | When a private reference directory is configured, the worker aligns the reference and calculates accuracy **after** decoding. Regression checks exercise renamed trips, removed references, renumbered roads and changed synthetic turns. | Saved server checks cover the ten student-test trips. The UI receives aggregate agreement while reference labels stay on the server. |

Two details matter when explaining the system. First, the saved classifier timings cover scoring **already prepared features**. The API measurement includes worker startup, input loading, feature generation, decoding and optional scoring. Second, changing the data or model requires manually rebuilding the corresponding artifacts. The repository’s experiment records and request logs support reproducibility and diagnosis. Continuous retraining, a hosted model registry and drift alerts are outside the implemented workflow. [API](pathsnap/api.py) · [Inference worker](pathsnap/serve_worker.py) · [Prediction checks](pathsnap/checks.py) · [API timings](results/api_checks.csv)
