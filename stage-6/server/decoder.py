"""Directed-road HMM with observation-dependent transitions and log-space Viterbi."""

import numpy as np
from scipy.special import logsumexp

from geometry import local_metres


def road_transitions(track,nodes,arcs,ids,features):
    """Prepare transition topology and distances without access to reference routes."""
    origin=track[:,:2].mean(axis=0)
    xy=local_metres(nodes,origin)
    gps=local_metres(track[:,:2],origin)
    n=len(nodes)
    keys=arcs[:,0].astype(np.int64)*n+arcs[:,1]
    sort=np.argsort(keys,kind="stable")
    keys=keys[sort]
    valid=ids>=0
    safe=np.maximum(ids,0)
    prev=safe[:-1,:,None]
    curr=safe[1:,None,:]
    end=arcs[prev,1].astype(np.int64)
    start=arcs[curr,0]
    middle_pair=end*n+start
    index=np.minimum(np.searchsorted(keys,middle_pair),len(keys)-1)
    has_middle=keys[index]==middle_pair
    same=prev==curr
    direct=end==start
    bridge=has_middle & ~direct & ~same
    bridge_ids=np.where(bridge,sort[index],-1).astype(np.int32)
    middle_lengths=np.linalg.norm(xy[arcs[:,1]]-xy[arcs[:,0]],axis=1)
    middle=np.where(bridge,middle_lengths[sort[index]],0)
    prev_len=features[:-1,:,2,None]
    prev_pos=features[:-1,:,3,None]*prev_len
    next_pos=features[1:,None,:,3]*features[1:,None,:,2]
    progress=next_pos-prev_pos
    road_dist=np.where(same,np.abs(progress),prev_len-prev_pos+middle+next_pos)
    observed=np.linalg.norm(np.diff(gps,axis=0),axis=1)[:,None,None]
    dt=np.diff(track[:,2])
    feasible=(valid[:-1,:,None] & valid[1:,None,:] & (same|direct|bridge)
              & (road_dist <= np.maximum(observed+80,60*dt[:,None,None]+30)))
    backward=np.where(same,np.maximum(0,-progress-5),0)
    mismatch=np.abs(road_dist-observed)+backward
    reverse=(arcs[prev,0]==arcs[curr,1]) & direct & ~same
    penalty=.25*(~same)+.75*bridge+1.5*reverse
    return mismatch.astype(np.float32),penalty.astype(np.float32),feasible,bridge_ids,dt


def transition_log(mismatch,penalty,feasible,scale):
    score=np.where(feasible,-mismatch/scale-penalty,-np.inf)
    # One unobserved exit gets weight exp(-3). Its emission probability is zero.
    norm=np.logaddexp(logsumexp(score,axis=2,keepdims=True),-3.)
    return score-norm


def viterbi(ids,probs,log_transition,dt):
    """Return arc IDs, selected slots and run starts. Empty observations get -1."""
    valid=ids>=0
    if probs.shape!=ids.shape or np.any(probs<0) or not np.isfinite(probs).all():
        raise ValueError("Invalid emissions for candidate slots.")
    log_emit=np.full(ids.shape,-np.inf)
    supported=valid & (probs>0)
    log_emit[supported]=np.log(probs[supported])
    back=np.full(ids.shape,-1,np.int16)
    slots=np.full(len(ids),-1,np.int16)
    starts=np.zeros(len(ids),dtype=bool)
    end_states=[]
    previous=None
    for i,emission in enumerate(log_emit):
        if not np.isfinite(emission).any():
            if previous is not None:
                end_states.append((i-1,int(np.argmax(previous))))
            previous=None
            continue
        if previous is not None and 0<dt[i-1]<=30:
            scores=previous[:,None]+log_transition[i-1]
            predecessor=np.argmax(scores,axis=0)
            current=emission+scores[predecessor,np.arange(len(emission))]
            if np.isfinite(current).any():
                back[i]=predecessor
                previous=current-np.max(current)
                continue
        if previous is not None:
            end_states.append((i-1,int(np.argmax(previous))))
        starts[i]=True
        previous=emission-np.max(emission)
    if previous is not None:
        end_states.append((len(ids)-1,int(np.argmax(previous))))
    for i,state in end_states:
        while True:
            slots[i]=state
            if starts[i]:
                break
            state=back[i,state]
            i-=1
    arc=np.where(slots>=0,ids[np.arange(len(ids)),np.maximum(slots,0)],-1)
    return arc,slots,starts


def decode(prepared,ids,probs,scale):
    mismatch,penalty,feasible,bridge,dt=prepared
    logs=transition_log(mismatch,penalty,feasible,scale)
    pred,slots,starts=viterbi(ids,probs,logs,dt)
    connectors=np.full(len(pred),-1,np.int32)
    j=np.flatnonzero((~starts[1:]) & (slots[1:]>=0) & (slots[:-1]>=0))
    connectors[j+1]=bridge[j,slots[j],slots[j+1]]
    return pred,starts,connectors
