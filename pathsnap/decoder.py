"""Viterbi over candidate arcs with directed shortest-path transitions."""
import heapq
from functools import lru_cache
import numpy as np
from scipy.special import logsumexp
from .core import projection

def decode_path(net,ids,x,emissions,config, forced_starts=None):
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
    valid_step = (dt > 0) & (dt <= config['gap_s'])
    travel_time = dt
    if forced_starts is not None:
        # Elapsed time between retained observations includes dwell time.
        # Preserve true sampling breaks and bound the search through long stops.
        valid_step = dt > 0
        valid_step[np.asarray(forced_starts[1:], dtype=int) - 1] = False
        travel_time = np.minimum(dt, config['gap_s'])
    bounds=np.maximum(observed,travel_time*config['speed_mps'])+3*scale
    budgets=np.ceil(bounds/scale)*scale
    log_e=np.where(emissions > 0, np.log(np.maximum(emissions,1e-300)), -np.inf)
    reversal_cost = config.get("reversal_cost", 0.0)
    if config.get("motion_prior", False):
        # Neighbouring feature windows share GPS samples. Temper repeated classifier
        # evidence and retain a direct distance likelihood for the observed point.
        log_e = 0.5 * log_e - np.expm1(x[:, :, 0]) / scale
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
        if valid_step[i-1]:
            middle=np.array([[paths(a,budgets[i-1])[0].get(b,np.inf) for b in edges[i,:,0]]
                             for a in edges[i-1,:,1]])
            travel=(net.length[ids[i-1]]-along[i-1])[:,None]+middle+along[i]
            offset=along[i][None,:]-along[i-1][:,None]
            direct=(ids[i-1,:,None]==ids[i,None,:])&(offset>=-scale)
            travel=np.where(direct,np.minimum(travel,np.maximum(offset,0)),travel)
            transition=np.where(travel<=bounds[i-1],-abs(travel-observed[i-1])/scale,-np.inf)
            if config.get("motion_prior", False):
                reversed_edge = (
                    (edges[i - 1, :, 0, None] == edges[i, None, :, 1])
                    & (edges[i - 1, :, 1, None] == edges[i, None, :, 0])
                )
                transition -= reversal_cost * reversed_edge
            else:
                norm = logsumexp(transition, axis=1, keepdims=True)
                transition -= np.where(np.isfinite(norm), norm, 0)
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
    anchors = np.zeros(len(ids), dtype=int)
    for begin,end in zip(starts,starts[1:]+[len(ids)]):
        fragment=[int(pred[begin])]
        for i in range(begin+1,end):
            prev,curr=pred[i-1:i+1]
            if prev==curr and along[i,choice[i]]>=along[i-1,choice[i-1]]-scale:
                anchors[i] = len(fragment) - 1
                continue
            source,target=net.arcs[prev,1],net.arcs[curr,0]
            parent=paths(source,budgets[i-1])[1]
            connector=[]
            while target!=source:
                target,arc=parent[target];connector.append(arc)
            fragment.extend(reversed(connector));fragment.append(int(curr))
            anchors[i] = len(fragment) - 1
        fragments.append(fragment)
    return pred, fragments, starts, anchors


def align_to_prediction(net, fragments, starts, anchors, radius):
    """Place observations in traversal order on a local window of the inferred path."""
    result = np.empty(len(net.gps), dtype=int)
    for route, first, end in zip(fragments, starts, starts[1:] + [len(result)]):
        route = np.asarray(route)
        width = min(2 * radius + 1, len(route))
        left = np.clip(anchors[first:end] - radius, 0, len(route) - width)
        columns = np.arange(width)
        occurrences = left[:, None] + columns
        distance, _ = projection(net, net.gps[first:end, None], route[occurrences])
        cost = distance[0] ** 2
        back = np.empty(distance.shape, dtype=int)
        for i in range(1, end - first):
            # The left edge only moves forward. A predecessor can be any previous
            # occurrence up to and including the current occurrence.
            limit = np.minimum(columns + left[i] - left[i - 1], width - 1)
            prefix = np.minimum.accumulate(cost)
            best = np.maximum.accumulate(np.where(cost == prefix, columns, 0))
            back[i] = best[limit]
            cost = prefix[limit] + distance[i] ** 2
        col = int(cost.argmin())
        for i in reversed(range(end - first)):
            result[first + i] = route[left[i] + col]
            if i:
                col = back[i, col]
    return result


def decode(net, ids, x, emissions, config):
    pred, fragments, starts, anchors = decode_path(net, ids, x, emissions, config)
    radius = config.get("alignment_radius", 0)
    if radius:
        pred = align_to_prediction(net, fragments, starts, anchors, radius)
    return pred, fragments, starts
