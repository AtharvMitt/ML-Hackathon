"""Per-request worker. Only the post-prediction scoring step can load a reference."""
from pathlib import Path
import argparse
import json
import joblib
from .predict import predict
from .reference import align,read_reference

def response(directory,artifacts,reference_root):
    bundle=joblib.load(artifacts/'model.joblib')
    config=json.loads((artifacts/'decoder.json').read_text())
    net,pred,fragments,starts=predict(directory,bundle,config)
    lines=[]
    for arcs in fragments:
        # Connector arcs are contiguous inside each decoded fragment.
        endpoints=net.arcs[arcs]
        lines.append([net.nodes[endpoints[0,0]].tolist(),*net.nodes[endpoints[:,1]].tolist()])
    result=dict(trip=directory.name,track=net.track[:,:2].tolist(),route=lines,
                observation_arcs=pred.tolist(),fragment_starts=starts,metrics=None)
    if reference_root is not None:
        reference_dir=reference_root/directory.name
        if any(reference_dir.glob('*.route')):
            truth=align(net,read_reference(reference_dir,net))
            result['metrics']=dict(observations=len(pred),top1=float((pred==truth).mean()))
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('directory',type=Path)
    p.add_argument('--artifacts',type=Path,required=True)
    p.add_argument('--reference-root',type=Path)
    args=p.parse_args()
    print(json.dumps(response(args.directory,args.artifacts,args.reference_root),separators=(',',':')))
