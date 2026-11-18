"""Offline reference alignment. This module is never needed to predict."""
import numpy as np
from .core import projection

def read_reference(directory, net):
    path, = directory.glob("*.route")
    route = np.loadtxt(path,dtype=int,ndmin=1)
    if route.ndim != 1 or not len(route) or route.min()<0 or route.max()>=len(net.arcs):
        raise ValueError("Invalid route")
    return route

def align(net, route):
    """Minimum total squared point-to-segment residual under ordered occurrences."""
    positions = np.arange(len(route))
    cost = np.zeros(len(route))
    back = np.empty((len(net.gps),len(route)),np.int32)
    for first in range(0,len(net.gps),128):
        distances, _ = projection(net,net.gps[first:first+128,None],route[None,:])
        for i, residual in enumerate(distances,first):
            prefix = np.minimum.accumulate(cost)
            back[i] = np.maximum.accumulate(np.where(cost == prefix,positions,0))
            cost = prefix + residual**2
    pos = int(cost.argmin())
    aligned = np.empty(len(net.gps),int)
    for i in reversed(range(len(aligned))):
        aligned[i] = route[pos]
        pos = back[i,pos]
    return aligned
