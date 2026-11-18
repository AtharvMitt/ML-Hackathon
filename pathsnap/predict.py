import argparse
from pathlib import Path
from dataclasses import replace
import numpy as np
import json
import joblib
from .core import features,read_trip,FEATURES
from .decoder import decode_path, align_to_prediction
from .experiments import probabilities

def smooth_gps(net, gap):
    """Local linear position estimates. Isolate gaps and preserve constant velocity."""
    gps = net.gps.copy()
    time = net.track[:, 2]
    breaks = np.flatnonzero((np.diff(time) <= 0) | (np.diff(time) > gap)) + 1
    for rows in np.split(np.arange(len(time)), breaks):
        if len(rows) < 3:
            continue
        start, end = rows[0], rows[-1]
        neighbours = rows[:, None] + np.arange(-4, 5)
        valid = (neighbours >= start) & (neighbours <= end)
        neighbours = np.clip(neighbours, start, end)
        offsets = time[neighbours] - time[rows, None]
        bandwidth = 2 * np.median(np.diff(time[rows]))
        weights = np.exp(-0.5 * (offsets / bandwidth) ** 2) * valid
        # Intercept of a weighted linear fit at each observation time.
        t1 = (weights * offsets).sum(axis=1, keepdims=True)
        t2 = (weights * offsets**2).sum(axis=1, keepdims=True)
        intercept = weights * (t2 - t1 * offsets)
        total = intercept.sum(axis=1, keepdims=True)
        stable = total[:, 0] > 1e-12
        weighted_xy = intercept[stable, :, None] * net.gps[neighbours[stable]]
        gps[rows[stable]] = weighted_xy.sum(axis=1) / total[stable]
    return replace(net, gps=gps)


def motion_anchors(net, config):
    """Retain endpoints and spatially distinct observations within each sampling fragment."""
    time = net.track[:, 2]
    intervals = np.diff(time)
    breaks = np.flatnonzero((intervals <= 0) | (intervals > config["gap_s"])) + 1
    keep = []
    forced_starts = []
    for rows in np.split(np.arange(len(time)), breaks):
        forced_starts.append(len(keep))
        last = rows[0]
        keep.append(last)
        for i in rows[1:-1]:
            if np.linalg.norm(net.gps[i] - net.gps[last]) >= 2 * config["distance_scale_m"]:
                keep.append(i)
                last = i
        if rows[-1] != last:
            keep.append(rows[-1])
    return np.asarray(keep), np.asarray(forced_starts)


def search_route(net, model, k, config):
    search_net = smooth_gps(net, config["gap_s"]) if config.get("smooth_gps") else net
    width = 2 * k if config.get("wide_search") else k
    ids, x = features(search_net, width)
    scores = probabilities(model, x)
    if config.get("wide_search") and ids.shape[1] > k:
        # Keep all original candidates. Admit farther roads when the closest road
        # fits poorly or the extra road is within one spatial error scale of it.
        dist = np.expm1(x[:, :, 0])
        scale = config["distance_scale_m"]
        plausible = (dist[:, :1] > scale) | (dist[:, k:] <= dist[:, :1] + scale)
        scores[:, k:] = np.where(plausible, scores[:, k:], 0)
    if config.get("use_anchors"):
        keep, forced = motion_anchors(search_net, config)
        sparse = replace(search_net, track=search_net.track[keep], gps=search_net.gps[keep])
        _, fragments, starts, anchors = decode_path(
            sparse, ids[keep], x[keep], scores[keep], config, forced_starts=forced
        )
        # Map retained path positions back to original sample indices within each
        # fragment. The final ordered alignment below uses every original GPS point.
        starts = keep[starts].tolist()
        dense = np.zeros(len(net.gps), dtype=int)
        for first, end in zip(starts, starts[1:] + [len(dense)]):
            inside = (keep >= first) & (keep < end)
            positions = np.interp(np.arange(first, end), keep[inside], anchors[inside])
            dense[first:end] = np.rint(positions).astype(int)
        path = align_to_prediction(net, fragments, starts, dense, config.get("alignment_radius") or k)
        return path, fragments, starts
    path, fragments, starts, anchors = decode_path(search_net, ids, x, scores, config)
    radius = config.get("alignment_radius", 0)
    if radius:
        path = align_to_prediction(net, fragments, starts, anchors, radius)
    return path, fragments, starts


def predict(directory,bundle,config):
    if tuple(bundle['features'])!=FEATURES: raise ValueError('Model feature schema mismatch')
    net=read_trip(directory)
    pred,segments,breaks=search_route(net, bundle["model"], bundle["k"], config)
    return net, pred, segments, breaks

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('trip',type=Path)
    args=parser.parse_args()
    _,pred,segments,starts=predict(args.trip,joblib.load('artifacts/model.joblib'),
        json.loads(Path('artifacts/decoder.json').read_text()))
    print(json.dumps(dict(observation_arcs=pred.tolist(),route_fragments=segments,fragment_starts=starts)))
