"""Viterbi over candidate arcs with directed shortest-path transitions."""
import heapq
from functools import lru_cache
import numpy as np
from scipy.special import logsumexp

def decode(net,ids,x,emissions,config):
    adjacency=[[] for _ in net.nodes]
    for arc,((a,b),length) in enumerate(zip(net.arcs,net.length)):
        if length>0: adjacency[a].append((b,length,arc))
    scale=config['distance_scale_m']
    @lru_cache(maxsize=8192)
    def paths(source,budget):
        distances={source:0.}
        parent={}
        queue=[(0.,source)]
        while queue:
            d,node=heapq.heappop(queue)
            if d>distances[node]: continue
            for target,length,arc in adjacency[node]:
                cost=d+length
                if cost<=budget and cost<distances.get(target,np.inf):
                    distances[target]=cost
                    parent[target]=(node,arc)
                    heapq.heappush(queue,(cost,target))
        return distances,parent
    dt=np.diff(net.track[:,2])
    observed=np.linalg.norm(np.diff(net.gps,axis=0),axis=1)
    # Round cache radii upward for reuse; mask distances against the true budget.
    bounds=np.maximum(observed,dt*config['speed_mps'])+3*scale
    budgets=np.ceil(bounds/scale)*scale
    log_e=np.log(np.maximum(emissions,1e-300))
    values=log_e[0].copy()
    back=np.full(ids.shape,-1,int)
    choice=np.empty(len(ids),int)
    starts=[0]
    along=net.length[ids]*x[:,:,2]
    edges=net.arcs[ids]
    def trace(start,end,scores):
        choice[end]=scores.argmax()
        for j in range(end,start,-1):
            choice[j-1]=back[j,choice[j]]
    for i in range(1,len(ids)):
        delta=dt[i-1]
        if 0<delta<=config['gap_s']:
            middle=np.array([[paths(a,budgets[i-1])[0].get(b,np.inf) for b in edges[i,:,0]]
                             for a in edges[i-1,:,1]])
            travel=(net.length[ids[i-1]]-along[i-1])[:,None]+middle+along[i]
            offset=along[i][None,:]-along[i-1][:,None]
            direct=(ids[i-1,:,None]==ids[i,None,:])&(offset>=-scale)
            travel=np.where(direct,np.minimum(travel,np.maximum(offset,0)),travel)
            transition=np.where(travel<=bounds[i-1],-abs(travel-observed[i-1])/scale,-np.inf)
            norm=logsumexp(transition,axis=1,keepdims=True)
            transition-=np.where(np.isfinite(norm),norm,0)
            scores=values[:,None]+transition
            next_values=scores.max(axis=0)+log_e[i]
            if np.isfinite(next_values).any():
                back[i]=scores.argmax(axis=0);values=next_values
                continue
        trace(starts[-1],i-1,values)
        starts.append(i);values=log_e[i].copy()
    trace(starts[-1],len(ids)-1,values)
    pred=ids[np.arange(len(ids)),choice]
    fragments=[]
    for begin,end in zip(starts,starts[1:]+[len(ids)]):
        fragment=[int(pred[begin])]
        for i in range(begin+1,end):
            prev,curr=pred[i-1:i+1]
            if prev==curr and along[i,choice[i]]>=along[i-1,choice[i-1]]-scale: continue
            source,target=net.arcs[prev,1],net.arcs[curr,0]
            parent=paths(source,budgets[i-1])[1]
            connector=[]
            while target!=source:
                target,arc=parent[target];connector.append(arc)
            fragment.extend(reversed(connector));fragment.append(int(curr))
        fragments.append(fragment)
    return pred,fragments,starts
