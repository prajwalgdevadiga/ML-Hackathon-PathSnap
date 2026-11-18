"""Inference entry point. Call with a fitted estimator, decoder scale and three raw arrays."""
from time import perf_counter

from decoder import road_transitions,decode
from candidates import generate


def predict_trip(model,scale,track,nodes,arcs):
    from shared.compare import predict, columns
    t0=perf_counter()
    ids,features,fallback=generate(track,nodes,arcs)
    ranked,prob=predict(model,columns("hist_boost"),features,ids)
    geometry=road_transitions(track,nodes,arcs,ids,features)
    pred,starts,connectors=decode(geometry,ids,prob,scale)
    return dict(classifier=ranked[:,0],decoded=pred,starts=starts,connectors=connectors,
                fallback=fallback,seconds=perf_counter()-t0)
