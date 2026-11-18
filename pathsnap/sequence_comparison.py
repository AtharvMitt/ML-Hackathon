"""Reproduce whole-trip metrics for the best validated setting of each feature set.

Run with python -m pathsnap.sequence_comparison from the repository root.
"""

import argparse
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from threadpoolctl import threadpool_limits

from .temporal import choose_arcs
from .temporal_training import fit_bundle, fit_ranker, read_candidates


def evaluate(trips, model, variant, split, fold):
    rows = []
    for name, trip in trips.items():
        ids, edges = trip["ids"], trip["edges"]
        start = perf_counter()
        pred = choose_arcs(model, ids, edges, trip["x"], trip["t"])
        elapsed = perf_counter() - start
        matches = ids == pred[:, None]
        assert matches.any(axis=1).all()
        endpoints = edges[np.arange(len(ids)), matches.argmax(axis=1)]
        continuous = (pred[1:] == pred[:-1]) | (endpoints[:-1, 1] == endpoints[1:, 0])
        truth = trip["y"]
        rows.append(
            {
                "variant": variant,
                "partition": split,
                "fold": fold,
                "trip": name,
                "n": len(truth),
                "correct": int((truth == pred).sum()),
                "macro_f1": f1_score(truth, pred, average="macro", zero_division=0),
                "continuity": continuous.mean(),
                "seconds": elapsed,
            }
        )
    return rows


def load(directory, cache):
    trips = {}
    for file in sorted(directory.glob("*.csv")):
        cached = cache / f"graph-{directory.name}{file.stem}.npz" if cache else None
        if cached and cached.exists():
            with np.load(cached) as data:
                trips[file.stem] = dict(data)
        else:
            trips[file.stem] = read_candidates(file)
    return trips


def main(cache=None):
    run = Path("notebook_run")
    train = load(Path("cleaned_data/train"), cache)
    test = load(Path("cleaned_data/student_test"), cache)
    folds = pd.read_csv("results/folds.csv", dtype={"trip": str}).set_index("trip").fold
    cv = pd.read_csv(run / "temporal_cv.csv").fillna({"align": -1})
    cv["accuracy"] = cv.correct / cv.total
    settings = ["variant", "cap", "iterations", "weight", "align"]
    ranked = cv.groupby(settings).accuracy.mean().sort_values(ascending=False)
    rows = []
    # Feature families append channels to the original geometry in this order.
    for variant, width in [
        ("basic", 35),
        ("extended", 59),
        ("graph", 68),
        ("stack", 68),
    ]:
        _, cap, iterations, weight, strength = ranked.xs(
            variant, drop_level=False
        ).idxmax()
        for fold in [*sorted(folds.unique()), -1]:
            training = {
                k: dict(t, x=t["x"][:, :, :width])
                for k, t in train.items()
                if fold < 0 or folds[k] != fold
            }
            held = (
                test
                if fold < 0
                else {k: t for k, t in train.items() if folds[k] == fold}
            )
            held = {k: dict(t, x=t["x"][:, :, :width]) for k, t in held.items()}
            alignment = None if strength < 0 else strength
            if variant == "stack":
                model = (
                    joblib.load("artifacts/temporal.joblib")
                    if fold < 0
                    else fit_bundle(training, variant, weight, alignment)
                )
            else:
                model = {
                    "model": fit_ranker(training, cap, iterations),
                    "weight": weight,
                    "alignment": alignment,
                }
            rows.extend(
                evaluate(
                    held, model, variant, "student_test" if fold < 0 else "cv", fold
                )
            )
        print(f"Scored {variant}", flush=True)
    pd.DataFrame(rows).to_csv(run / "sequence_comparison.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache", type=Path, help="Optional directory of precomputed input features."
    )
    with threadpool_limits(limits=3):
        main(parser.parse_args().cache)
