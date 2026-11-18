from pathlib import Path
import argparse
import json
from time import perf_counter
from itertools import groupby
import numpy as np
import joblib
from sklearn.metrics import f1_score
from .core import read_trip,projection
from .decoder import decode
from .experiments import probabilities,save_csv

def edit(a,b):
    # Vectorised row update for unit insertion, deletion, substitution cost.
    b=np.asarray(b)
    previous=np.arange(len(b)+1)
    index=previous.copy()
    for i,item in enumerate(a,1):
        row=np.r_[i,np.minimum(previous[1:]+1,previous[:-1]+(b!=item))]
        previous=np.minimum.accumulate(row-index)+index
    return previous[-1]/max(len(a),len(b),1)

def collapsed(seq):
    return [k for k,_ in groupby(seq)]
def run(root,cache):
    residuals=[]; intervals=[]; speeds=[]
    for file in sorted((cache/'train').glob('*.npz')):
        net=read_trip(root/'train'/file.stem)
        with np.load(file) as data:
            residuals.extend(projection(net,net.gps,data['truth'])[0])
        dt=np.diff(net.track[:,2])
        intervals.extend(dt[dt>0])
        speeds.extend(np.linalg.norm(np.diff(net.gps,axis=0),axis=1)[dt>0]/dt[dt>0])
    config=dict(distance_scale_m=float(np.quantile(residuals,.75)),gap_s=float(np.quantile(intervals,.995)),
                speed_mps=float(np.quantile(speeds,.99)))
    Path('artifacts/decoder.json').write_text(json.dumps(config,indent=2))
    model=joblib.load('artifacts/model.joblib')['model']
    rows=[]
    for file in sorted((cache/'student_test').glob('*.npz')):
        net=read_trip(root/'student_test'/file.stem)
        with np.load(file) as data:
            ids,x,truth,route=(data[k] for k in ('ids','x','truth','route'))
        p=probabilities(model,x)
        raw=ids[np.arange(len(ids)),p.argmax(axis=1)]
        start=perf_counter()
        pred,segments,breaks=decode(net,ids,x,p,config)
        seconds=perf_counter()-start
        path=[]
        for segment in segments:
            if path: path.append(-1) # count a fragment break in sequence edit distance
            path.extend(collapsed(segment))
        for label,obs,sequence in [('classifier',raw,collapsed(raw)),('viterbi',pred,path)]:
            sequence=np.asarray(sequence)
            a,b=sequence[:-1],sequence[1:]
            ok=(a>=0)&(b>=0)
            join=np.zeros(len(ok),bool)
            join[ok]=net.arcs[a[ok],1]==net.arcs[b[ok],0]
            rows.append(dict(trip=file.stem,method=label,n=len(obs),top1=np.mean(obs==truth),
                macro_f1=f1_score(truth,obs,average='macro',zero_division=0),
                normalized_edit=edit(sequence,collapsed(route)),route_arc_continuity=join.mean() if len(join) else 1.,
                fragments=len(segments) if label=='viterbi' else '',decoder_seconds=seconds if label=='viterbi' else 0))
        print(file.stem,flush=True)
    save_csv('route_comparison.csv',rows)
if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--data',type=Path,default=Path('Student folder'))
    p.add_argument('--cache',type=Path,default=Path('.cache/features'))
    args=p.parse_args()
    run(args.data,args.cache)
