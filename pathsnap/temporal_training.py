"""Train the trip-context ranker from the project's exported candidate tables."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from .core import DEFAULT_K, FEATURES
from .temporal import choose_arcs, geometry_features, rank_scores, score_context


def read_candidates(file):
    frame = pd.read_csv(file).sort_values(["observation", "candidate_rank"])
    groups = frame.groupby("observation", sort=False)
    widths = groups.size()
    if (
        widths.empty
        or widths.min() != widths.max()
        or frame.duplicated(["observation", "candidate_rank"]).any()
    ):
        raise ValueError(f"Incomplete candidate table in {file}")
    shape = (len(widths), int(widths.iloc[0]))
    ids = frame.candidate_arc_id.to_numpy().reshape(shape)
    edges = frame[["from_node", "to_node"]].to_numpy().reshape(*shape, 2)
    base = frame[list(FEATURES)].to_numpy("float32").reshape(*shape, len(FEATURES))
    obs = frame.drop_duplicates("observation")
    times = obs.timestamp_s.to_numpy()
    gps = obs[["longitude", "latitude"]].to_numpy()
    return {
        "ids": ids,
        "edges": edges,
        "t": times,
        "x": geometry_features(ids, edges, base, times, gps),
        "y": obs.reference_arc_id.to_numpy(),
    }


def fit_ranker(trips, cap, iterations):
    rng = np.random.default_rng(41)
    features, labels = [], []
    for trip in trips.values():
        matches = trip["ids"] == trip["y"][:, None]
        eligible = np.flatnonzero(matches.any(axis=1))
        if len(eligible) > cap:
            eligible = rng.choice(eligible, cap, replace=False)
        x = trip["x"]
        features.append(x[eligible].reshape(-1, x.shape[-1]))
        labels.append(matches[eligible].ravel())
    return HistGradientBoostingClassifier(
        learning_rate=0.075,
        max_iter=iterations,
        max_leaf_nodes=31,
        min_samples_leaf=40,
        early_stopping=False,
        random_state=41,
    ).fit(np.concatenate(features), np.concatenate(labels))


def fit_bundle(trips, variant, weight, alignment):
    """Fit the second stage on first-stage predictions from unseen trips."""
    if variant != "stack":
        raise ValueError(f"Expected the frozen two-stage model, got {variant}")
    names = sorted(trips)
    np.random.default_rng(41).shuffle(names)
    learned = {}
    for subset in np.array_split(names, 3):
        held = set(subset)
        first = fit_ranker({k: v for k, v in trips.items() if k not in held}, 768, 120)
        for name in subset:
            trip = trips[name]
            x = trip["x"]
            scores = rank_scores(first, x)
            learned[name] = dict(
                trip, x=score_context(trip["ids"], trip["edges"], x, trip["t"], scores)
            )
    second = fit_ranker(learned, 1536, 200)
    return {
        "kind": "temporal",
        "variant": variant,
        "weight": weight,
        "alignment": alignment,
        "k": DEFAULT_K,
        "features": FEATURES,
        "model": fit_ranker(trips, 768, 120),
        "second_stage": second,
    }


def score_trip(trip_id, prediction, truth):
    if prediction.shape != truth.shape or not len(truth):
        raise ValueError("Expected one prediction per observation")
    correct = int(np.count_nonzero(prediction == truth))
    return {
        "trip": trip_id,
        "correct": correct,
        "total": len(truth),
        "accuracy": correct / len(truth),
    }


def score_candidates(directory, bundle):
    rows = []
    for file in sorted(Path(directory).glob("*.csv")):
        trip = read_candidates(file)
        pred = choose_arcs(bundle, trip["ids"], trip["edges"], trip["x"], trip["t"])
        rows.append(score_trip(file.stem, pred, trip["y"]))
    return rows


def score_archive(archive, models, config):
    """Predict from network and GPS inputs before reading each private reference."""
    from tempfile import TemporaryDirectory
    from zipfile import ZipFile

    from .predict import predict
    from .reference import align, read_reference

    rows = []
    with ZipFile(archive) as source, TemporaryDirectory() as temporary:
        tracks = sorted(
            Path(n)
            for n in source.namelist()
            if n.endswith(".track") and "__MACOSX" not in Path(n).parts
        )
        for track in tracks:
            trip = track.stem
            directory = Path(temporary) / trip
            directory.mkdir()
            for suffix in (".track", ".nodes", ".arcs"):
                (directory / (trip + suffix)).write_bytes(
                    source.read(str(track.with_suffix(suffix)))
                )
            outputs = {}
            for name, bundle in models.items():
                network, outputs[name], _, _ = predict(directory, bundle, config)
            (directory / (trip + ".route")).write_bytes(
                source.read(str(track.with_suffix(".route")))
            )
            truth = align(network, read_reference(directory, network))
            for name, prediction in outputs.items():
                rows.append(dict(score_trip(trip, prediction, truth), model=name))
    return rows
