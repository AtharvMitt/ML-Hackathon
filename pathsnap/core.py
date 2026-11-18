"""Geometry and candidate features shared by training and live inference."""
from dataclasses import dataclass
import numpy as np
from scipy.spatial import cKDTree

FEATURES = ("log_distance", "distance_above_best", "along_fraction", "log_length",
            "incoming_alignment", "outgoing_alignment", "window_alignment",
            "incoming_speed", "outgoing_speed", "incoming_overshoot", "outgoing_overshoot")
DEFAULT_K = 8

@dataclass
class NetworkTrack:
    track: np.ndarray
    nodes: np.ndarray
    arcs: np.ndarray
    gps: np.ndarray
    start: np.ndarray
    vector: np.ndarray
    length: np.ndarray

def read_trip(directory):
    def load(ext, dtype):
        path, = directory.glob("*." + ext)
        return np.loadtxt(path, dtype=dtype, ndmin=2)
    track, nodes, arcs = load("track", float), load("nodes", float), load("arcs", int)
    if track.shape[1] != 3 or nodes.shape[1] != 2 or arcs.shape[1] != 2:
        raise ValueError("Invalid track or network columns")
    if not len(track) or not len(arcs) or not np.isfinite(track).all() or not np.isfinite(nodes).all():
        raise ValueError("Missing or nonfinite track/network")
    if arcs.min() < 0 or arcs.max() >= len(nodes) or np.any(np.diff(track[:, 2]) < 0):
        raise ValueError("Invalid node index or unordered track")
    origin = nodes.mean(axis=0)
    scale = np.pi / 180 * 6371008.8 * np.array([np.cos(np.deg2rad(origin[1])), 1.])
    xy, gps = (nodes - origin) * scale, (track[:, :2] - origin) * scale
    start = xy[arcs[:, 0]]
    vector = xy[arcs[:, 1]] - start
    return NetworkTrack(track, nodes, arcs, gps, start, vector, np.linalg.norm(vector, axis=1))

def projection(net, points, ids):
    offset = points - net.start[ids]
    along = (offset * net.vector[ids]).sum(axis=-1) / np.maximum(net.length[ids]**2, 1e-15)
    fraction = np.clip(along, 0, 1)
    distance = np.linalg.norm(offset - fraction[..., None] * net.vector[ids], axis=-1)
    return distance, fraction

def candidates(net, k, spacing=40.):
    """Midpoints of subsegments accelerate exact nearest-segment queries."""
    valid = np.flatnonzero(net.length > 0)
    if not len(valid):
        raise ValueError("No positive-length arcs")
    k = min(k, len(valid))
    pieces = np.maximum(1, np.ceil(net.length[valid] / spacing).astype(int))
    ids = np.repeat(valid, pieces)
    local = np.arange(len(ids)) - np.repeat(np.cumsum(pieces)-pieces, pieces)
    fraction = (local + .5) / np.repeat(pieces, pieces)
    tree = cKDTree(net.start[ids] + fraction[:, None] * net.vector[ids])
    midpoints = cKDTree(net.start[valid] + .5 * net.vector[valid])
    _, upper_ids = midpoints.query(net.gps, k=list(range(1,k+1)))
    upper = projection(net, net.gps[:,None], valid[upper_ids])[0].max(axis=1)
    # Triangle inequality gives a conservative bound for a witness midpoint.
    hits = tree.query_ball_point(net.gps, upper + spacing/2 + 1e-7)
    counts = np.fromiter(map(len, hits), int)
    obs = np.repeat(np.arange(len(hits)), counts)
    pair = np.unique(obs * len(net.arcs) + ids[np.concatenate(hits).astype(int)])
    obs, arcs = np.divmod(pair, len(net.arcs))
    distance, _ = projection(net, net.gps[obs], arcs)
    # Geometric secondary keys keep boundary ties independent of input row order.
    order = np.lexsort((net.vector[arcs,1], net.vector[arcs,0], net.start[arcs,1],
                       net.start[arcs,0], distance, obs))
    counts = np.bincount(obs)
    rank = np.arange(len(pair)) - np.repeat(np.cumsum(counts)-counts, counts)
    return arcs[order[rank < k]].reshape(-1,k)

def features(net, k=DEFAULT_K):
    ids = candidates(net,k)
    distance, fraction = projection(net, net.gps[:,None], ids)
    length = net.length[ids]
    unit = net.vector[ids] / length[:,:,None]
    idx = np.arange(len(net.track))
    def motion(a,b):
        delta = net.gps[b] - net.gps[a]
        norm = np.linalg.norm(delta,axis=1)
        cosine = (unit * delta[:,None,:]).sum(axis=2) / np.maximum(norm[:,None],1e-12)
        dt = net.track[b,2] - net.track[a,2]
        speed = np.divide(norm,dt,out=np.zeros_like(norm),where=dt>0)
        return cosine, np.broadcast_to(np.log1p(speed[:,None]),ids.shape), norm[:,None]
    cin, vin, din = motion(np.maximum(0,idx-1),idx)
    cout, vout, dout = motion(idx,np.minimum(idx+1,idx[-1]))
    wide, _, _ = motion(np.maximum(0,idx-3),np.minimum(idx+3,idx[-1]))
    x = np.stack((np.log1p(distance),np.log1p(distance-distance.min(axis=1,keepdims=True)),
                  fraction,np.log1p(length),cin,cout,wide,vin,vout,
                  np.log1p(np.maximum(0,din-fraction*length)),
                  np.log1p(np.maximum(0,dout-(1-fraction)*length))),axis=-1)
    return ids, x.astype("float32")
