"""Per-request worker. Only the post-prediction scoring step can load a reference."""
from pathlib import Path
import argparse
import json
import joblib
import numpy as np
from .predict import predict
from .reference import align,read_reference

def route_lines(net, arcs):
    """Split an ordered arc sequence wherever successive endpoints disconnect."""
    edges = net.arcs[arcs]
    breaks = np.flatnonzero(edges[:-1, 1] != edges[1:, 0]) + 1
    return [
        net.nodes[np.r_[part[0, 0], part[:, 1]]].tolist()
        for part in np.split(edges, breaks) if len(part)
    ]


def response(directory,artifacts,reference_root,show_reference=True):
    bundle=joblib.load(artifacts/'model.joblib')
    config=json.loads((artifacts/'decoder.json').read_text())
    net,pred,fragments,starts=predict(directory,bundle,config)
    lines = [line for fragment in fragments for line in route_lines(net, fragment)]
    # Opposite directions share one line on the display. Decoding uses the directed graph.
    street_edges = np.unique(np.sort(net.arcs, axis=1), axis=0)
    road_network = dict(nodes=net.nodes.tolist(), edges=street_edges.tolist())
    result=dict(trip=directory.name,track=net.track[:,:2].tolist(),route=lines,road_network=road_network,
                observation_arcs=pred.tolist(),fragment_starts=starts,metrics=None,reference_route=None)
    if reference_root is not None:
        reference_dir=reference_root/directory.name
        if any(reference_dir.glob('*.route')):
            reference = read_reference(reference_dir, net)
            truth = align(net, reference)
            if show_reference:
                result["reference_route"] = route_lines(net, reference)
            result['metrics']=dict(observations=len(pred),top1=float((pred==truth).mean()))
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('directory',type=Path)
    p.add_argument('--artifacts',type=Path,required=True)
    p.add_argument('--reference-root',type=Path)
    p.add_argument('--score-only', action='store_true')
    args=p.parse_args()
    print(json.dumps(response(args.directory,args.artifacts,args.reference_root,not args.score_only),separators=(',',':')))
