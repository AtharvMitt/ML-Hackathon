import argparse
from pathlib import Path
import csv
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from .core import read_trip, features, projection, DEFAULT_K
from .reference import read_reference, align

def prepare(root, output, k):
    seen = set()
    rows = []
    for split in ('train','student_test'):
        trips = sorted(p for p in (root/split).iterdir() if p.is_dir())
        if not trips:
            raise ValueError(f'Empty partition {split}')
        for directory in trips:
            if directory.name in seen:
                raise ValueError('Trip appears twice')
            seen.add(directory.name)
            net = read_trip(directory)
            route = read_reference(directory, net)
            truth = align(net, route)
            ids,x = features(net,k)
            d = projection(net,net.gps,truth)[0]
            dt = np.diff(net.track[:,2])
            graph = coo_matrix((np.ones(len(net.arcs)),net.arcs.T),shape=(len(net.nodes),)*2)
            rows.append(dict(split=split,trip=directory.name,observations=len(truth),nodes=len(net.nodes),
                arcs=len(net.arcs),route_steps=len(route),one_second_steps=int((dt==1).sum()),
                max_gap_s=dt.max(),duplicate_times=int((dt==0).sum()),missing_values=int((~np.isfinite(net.track)).sum()+(~np.isfinite(net.nodes)).sum()),
                west=net.track[:,0].min(),east=net.track[:,0].max(),south=net.track[:,1].min(),north=net.track[:,1].max(),
                components=connected_components(graph,directed=False,return_labels=False),
                isolated_nodes=len(net.nodes)-len(np.unique(net.arcs)),zero_length_arcs=int((net.length==0).sum()),
                reference_breaks=int((net.arcs[route[:-1],1]!=net.arcs[route[1:],0]).sum()),
                reference_distance_median_m=np.median(d),reference_distance_p95_m=np.quantile(d,.95),
                misses=int((~(ids==truth[:,None]).any(axis=1)).sum())))
            dest=output/split
            dest.mkdir(parents=True,exist_ok=True)
            np.savez(dest/(directory.name+'.npz'),x=x,ids=ids.astype('int32'),truth=truth,
                     route=route,edges=net.arcs[ids].astype('int32'))
        print(split,len(trips),flush=True)
    Path('results').mkdir(exist_ok=True)
    with open('results/inventory.csv','w') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--data',type=Path,default=Path('Student folder'))
    p.add_argument('--cache',type=Path,default=Path('.cache/features'))
    p.add_argument('--k',type=int,default=DEFAULT_K)
    args=p.parse_args()
    prepare(args.data,args.cache,args.k)
