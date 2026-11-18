import argparse
from pathlib import Path
import json
import joblib
from .core import features,read_trip,FEATURES
from .decoder import decode
from .experiments import probabilities

def predict(directory,bundle,config):
    if tuple(bundle['features'])!=FEATURES: raise ValueError('Model feature schema mismatch')
    net=read_trip(directory)
    ids,x=features(net,bundle['k'])
    p=probabilities(bundle['model'],x)
    pred,segments,breaks=decode(net,ids,x,p,config)
    return net, pred, segments, breaks

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('trip',type=Path)
    args=parser.parse_args()
    _,pred,segments,starts=predict(args.trip,joblib.load('artifacts/model.joblib'),
        json.loads(Path('artifacts/decoder.json').read_text()))
    print(json.dumps(dict(observation_arcs=pred.tolist(),route_fragments=segments,fragment_starts=starts)))
