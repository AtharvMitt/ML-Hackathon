# PathSnap hackathon plan

## What the team must deliver

Build a system that takes a time-ordered GPS track and its road network, identifies the directed road segment under each usable observation, and reconstructs the route between observations. The problem-statement video illustrates why this matters. Several roads can explain a single noisy point near a junction, while the observations before and after the junction can identify the turn the vehicle took.

The teacher requires **exactly two final submissions**.

1. **A public GitHub repository.** Include the code for data inspection, preprocessing, features, every training experiment, evaluation, Bayesian optimisation, HMM/Viterbi, Folium, FastAPI and the UI. Include reproducible dependencies, configurations, split definitions, seeds, trial logs, results, plotting scripts, saved model and inference assets subject to size limits, and a README for setup, training and two-laptop deployment. Keep credentials and reserved evaluation data out of the public repository.
2. **One consolidated technical-report PDF.** Follow sections 2–7 of Deliverables.pdf. Include partition-labelled EDA, every model tried, the comparison table, tuning results, decoder comparison, a Folium screenshot, the working UI and the successful API response on the receiving laptop's terminal. Every number and plot must trace to a real run. The team must independently implement and understand its work, be able to defend it in the viva, and disclose assistance as required.

**Live evaluation is part of the system.** Laptop 1 hosts FastAPI, the road networks, the model, the decoder and a separate private scorer. Laptop 2 lists the ten reserved trips and sends the selected ID to Laptop 1. Laptop 1 runs inference on request, scores the result privately, logs the request, and returns the predicted route and aggregate results. The evaluator can select any of the ten.

**Starting position.** The workspace has all five handouts, a ZIP containing those same PDFs, and the video. The raw trips, partition assignments and an official observation-to-arc scoring routine are absent. Obtain the teacher's training and student-testing package and check for a scoring specification before implementing labels. Keep the ten evaluator trips sealed during development. The following are proposed experiments and acceptance checks, with no claimed scores.

## Execution order

### 1. Establish the data and scoring contract

- Read the supplied partitions and discover files by trip. Verify the stated 80 training and 10 student-testing trips. Keep a manifest, checksums and stable observation indices. Group every split by entire trip. Reserve five reproducible folds of the 80 for cross-validation.
- Parse whitespace-delimited, headerless files. A track row is longitude, latitude and timestamp in seconds. A node row is longitude and latitude. An arc row contains its source and destination node indices. Node and arc IDs are zero-based row positions within that trip. Preserve those positions through cleaning. A route is an ordered list of arc IDs and may revisit an arc.
- Inventory sizes, coordinates, sampling intervals, duplicate timestamps, gaps, invalid rows, graph components, isolated nodes, invalid references and degenerate arcs. Verify directionality and coordinate order. Compute distances in metres using a local projection chosen from the track/network extent. Preserve an explicit status for unusable observations. Produce a training-trip Folium overlay and a partition-labelled EDA table.
- **Resolve supervision first.** Prefer official per-observation references or the teacher's scorer if supplied. If only ordered routes exist, build a separate offline reference aligner. Project each observation onto occurrences along the ordered reference path. Use dynamic programming to minimise total spatial residual with nondecreasing progress and a displacement-consistency cost. Allow dwell, repeated arcs and skipped short arcs. Record ties, coverage, residuals and inconsistent reference fragments. Inspect examples around junctions, gaps and loops before freezing the method. Label the resulting scores as agreement with derived references. Confirm scoring compatibility with the teacher. This aligner belongs only in training-label preparation and private scoring.
- Define denominators once. Report total, usable, reference-alignable and scored observations. Count a missing true candidate as a Top-1 error on every alignable observation. Report ambiguous/unscorable cases separately. Evaluate ordered route recovery against the supplied route itself as a second check independent of point alignment.

### 2. Generate candidates and learn a transferable road score

- Build a spatial index over directed segments for the current network. Retrieve nearby candidates using point-to-segment distance, preserving both directions where supplied. Expand the query or retrieve the nearest valid segment when the first query is empty. Expose fallback counts and distances. If the graph has no usable segments, return an explicit failure.
- On training folds measure candidate miss rate, candidates per point and runtime as the radius and candidate limit change. Select a common operating point on the training folds and freeze it for the model comparison. Candidate recall is the maximum Top-1 accuracy attainable by a classifier confined to that set. Training and inference must use the same generator. Never insert the reference arc into the candidate set.
- Make one example per observation–candidate pair. Features available at inference include perpendicular distance, relative distance to the next-best alternative, heading agreement with the directed arc, position along the segment, segment length, local connectivity, elapsed time and observed displacement. Mark heading as unavailable when motion is too small to estimate it reliably. Fit scaling, imputation, calibration and any learned tolerances only within the training portion of each fold.
- The target is whether that candidate equals the reference arc. Retain all realistic nearby alternatives, particularly reverse-direction and parallel segments. Train on alignable observations with a positive candidate and log excluded groups. Keep missing-positive groups in end-to-end evaluation. Give observations comparable training weight when candidate counts differ. Carry trip, observation and arc IDs as join keys outside the model feature matrix.

### 3. Compare five trained models before choosing one

Use a nearest-segment baseline to quantify the value of learning. Additionally train these five distinct models with the same candidates, features and folds.

| Model | Question the experiment answers |
|---|---|
| Scaled logistic regression | How far can additive distance and heading evidence take us? |
| Gaussian naive Bayes | Does a simple probabilistic model remain competitive despite correlated geometry features? |
| Decision tree | Do interactions between low motion, direction and distance improve decisions? |
| Extra Trees | Does averaging trees reduce fold-to-fold variability near junctions? |
| Histogram gradient boosting | Can successive corrections improve the ranking of close alternatives within the latency budget? |

- Convert candidate scores into a distribution within each observation, using the same documented transformation in evaluation and decoding. For example, apply a softmax to clipped binary log-odds. If calibrating its temperature, fit on grouped out-of-fold predictions from the fitting partition. Save predictions and timings for each run.
- Refit each baseline configuration on the 80 trips and report the required comparison on the ten student-testing trips. Select primarily by grouped CV observation-level Top-1 accuracy, which measures the actual road choice; use fold stability, log loss, decoder continuity and latency to interpret close results. Do not assume a winner. Retain the five per-model comparisons even if the simplest model wins.
- **Metric definitions.** Record candidate binary precision, recall, macro F1 and log loss separately from observation-level metrics. For observation-level macro F1, compare one predicted arc per point with its reference within each trip, macro-average across the union of observed and predicted arc classes, then average trips. An arc index means something different on another trip. Use the teacher's metric if specified and label both definitions if they differ. For observation-level log loss, report negative log probability of the reference on covered points alongside coverage and miss rate. Always include misses in overall Top-1. Record train, fold and student-test scores, train time, per-trip inference time, and conditions for the timing.
- Generate the required CV macro F1, student-test Top-1 and macro F1, and applicable log-loss plots from saved runs. Include confusion diagnostics where informative and epoch curves only for a model with actual epoch histories. For every model record its rationale, limitations, initial hyperparameters, objective and evidence of overfitting or underfitting.

### 4. Tune the selected family with Bayesian optimisation

Run a sequential Bayesian search, such as a Gaussian-process surrogate with expected improvement, over the chosen family's relevant parameters. If logistic regression wins, search regularisation strength on a log scale. If boosting wins, search learning rate, leaf count, minimum samples per leaf and regularisation. Set and record ranges and a trial/time budget from training-only pilots and available compute. Fix folds, seed and objective before starting. Maximise mean grouped CV Top-1 and record CV macro F1 for each trial. Retain trial parameters, scores, durations, failed trials and the best configuration. Refit on all 80 and compare untuned versus tuned on the same ten student-testing trips for macro F1, Top-1, log loss and inference time. Keep the measured outcome even if tuning fails to improve performance.

### 5. Decode and evaluate the whole route

- Each HMM state is a candidate directed arc with a projected position at time t. Use the classifier distribution for emissions. For each feasible transition, compute directed along-road travel distance between projected positions, including partial arcs and any connecting path. Handle forward travel along the same arc directly. Score the mismatch between network travel and measured GPS displacement using a residual scale estimated on training data. Account for elapsed time and train-derived speed plausibility. Normalise finite transition scores over reachable successors.
- Decode in log space. At each candidate, store its log emission plus the maximum previous score and log transition, with a backpointer to the winning predecessor. Backtrack and recover connector arcs as well as the arc under each observation. For a long gap, unavailable state or step with no reachable transition, record the break and restart from emissions. Draw separate fragments across breaks. Choose gap and transition settings on grouped development runs.
- Compare classifier-only and classifier-plus-Viterbi on identical trips and candidate sets. Report Top-1, per-trip macro F1, break counts, runtime, and the fraction of successive usable observations linked by a directionally feasible, time-plausible path, counting breaks as failures. For sequence quality, compare the traversed arc list with the ordered reference by normalised edit distance after collapsing consecutive duplicates consistently. Preserve later revisits. Document how connector arcs and breaks are counted. Inspect cases of both improvement and regression, especially parallel roads, turns after gaps and stopped vehicles.

### 6. Prove the demo runs without hardcoded answers

Package one inference function accepting only track, network, trained preprocessing/model and decoder configuration. Discover the trip catalog from the configured deployment partition. A trip ID selects input files. It never selects a model, a threshold or an answer. Generate the response and map geometry from the current request.

| Acceptance check on permitted development data | Passing evidence |
|---|---|
| Remove access to all route files from the predictor; run the same track again | Identical predictions. Only the separate scorer needs the reference. |
| Rename a trip, then consistently renumber its nodes and arcs | Same predicted geometry after mapping identifiers back, allowing documented geometric ties. |
| Send an unseen synthetic junction through the same API, then change its outgoing track direction | Live scores and decoded geometry respond to the new observations; the clean, unambiguous turn follows the changed track. |
| Run a stopped track, a gap and a disconnected graph | Stable observation indexing, finite scores where candidates exist, and explicit breaks or failure status. |
| Rehearse from Laptop 2 against Laptop 1 using a development catalog | Dynamic list, live request, server log with timing, predicted geometry, aggregate score and no reference geometry or labels in the response. |

Never use trip IDs, arc IDs, raw coordinate fingerprints, reference-derived features, cached predictions, hand-drawn routes, fixed demo metrics or per-trip branches to predict. File schemas, numerical safeguards, a fixed reproducibility seed and documented search budgets are implementation settings. Keep modelling choices configurable, record how they were selected, and load the frozen settings on both development and evaluator runs.

### 7. Schedule the demonstration and submission

- **Before lunch:** The handout announces a first-phase evaluation without specifying its rubric. Aim to show the data inventory, one verified training reference alignment, candidate coverage, a Folium overlay, a nearest-road baseline and one learned classifier running end to end. Start the five-model comparison early. This is a proposed checkpoint.
- **After initial comparisons:** Freeze the selected family, run Bayesian search, test decoder gains and rehearse two-laptop inference using student-testing trips. Build report tables directly from the saved results. For each model include family, configuration, CV macro F1, Top-1, route continuity, student-test macro F1, log loss and inference time. Explain any inapplicable entries.
- **Final evidence:** Generate a Folium map from a permitted development trip with toggleable GPS, network, reference and predicted layers, distinct styling, a legend and a fitted viewport. Save its screenshot and generation code. Capture the real UI and receiving terminal. Reproduce the result tables from a clean environment. Verify the public repo and single PDF agree with the demonstrated version. Use the reserved partition only at final evaluation, with private scoring after prediction.

## Source material

- [Main Guidelines](Main%20Guidelines.pdf) specifies the challenge and two-laptop architecture.
- [Dataset Exploration](Dataset%20Exploration.pdf) defines the file formats, partitions and EDA expectations.
- [Models and Training Guidelines](Models%20and%20Training%20Guidelines.pdf) specifies experimentation, tuning, decoding and final evaluation.
- [Deliverables](Deliverables.pdf) gives the submission checklist, report sections and before-lunch checkpoint.
- [Assessment and Plagiarism Policy](Assessment%20and%20Plagiarism%20Policy.pdf) defines independent work, evidence and evaluation integrity.
- [Problem Statement video](Problem%20Statement.mp4) visually illustrates GPS noise, competing roads and recovery of a connected route.
