import argparse
from pathlib import Path
import json
import joblib
import numpy as np
from .core import features,read_trip,FEATURES
from .decoder import decode
from .experiments import probabilities

def load_model(directory):
    """Use the sequence model when the artifact directory includes it."""
    sequence = directory / "temporal.joblib"
    return joblib.load(sequence if sequence.is_file() else directory / "model.joblib")


def predict(directory,bundle,config):
    if tuple(bundle['features'])!=FEATURES: raise ValueError('Model feature schema mismatch')
    net=read_trip(directory)
    ids,x=features(net,bundle['k'])
    if bundle.get("kind") == "temporal":
        from .temporal import choose_arcs, geometry_features
        edges = net.arcs[ids]
        times = net.track[:, 2]
        context = geometry_features(ids, edges, x, times, net.track[:, :2])
        chosen = choose_arcs(bundle, ids, edges, context, times)
        ranks = (ids == chosen[:, None]).argmax(axis=1)
        rows = np.arange(len(ids))
        ids = chosen[:, None]
        x = x[rows, ranks][:, None, :]
        p = np.ones(ids.shape)
    else:
        p=probabilities(bundle['model'],x)
    pred,segments,breaks=decode(net,ids,x,p,config)
    return net, pred, segments, breaks

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('trip',type=Path)
    args=parser.parse_args()
    _,pred,segments,starts=predict(args.trip,load_model(Path('artifacts')),
        json.loads(Path('artifacts/decoder.json').read_text()))
    print(json.dumps(dict(observation_arcs=pred.tolist(),route_fragments=segments,fragment_starts=starts)))
