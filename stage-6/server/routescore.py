import numpy as np

def lcs_length(first, second):
    # Each bit represents one reference position; update one full LCS row per token.
    masks = {}
    for j, token in enumerate(second):
        masks[token] = masks.get(token, 0) | (1 << j)
    state = 0
    for token in first:
        choices = masks.get(token, 0) | state
        state = choices & ~(choices - ((state << 1) | 1))
    return state.bit_count()

def metrics(track, arcs, route, labels, eligible, ids, fallback):
    pred = ids[:,0]
    y, h = labels[eligible], pred[eligible]
    classes = np.union1d(y, h)
    idx = np.searchsorted(classes, y)
    actual = np.bincount(idx, minlength=len(classes))
    guessed = np.bincount(np.searchsorted(classes, h), minlength=len(classes))
    tp = np.bincount(idx[y == h], minlength=len(classes))
    matched = ids[eligible] == y[:,None]
    ok_step = np.diff(track[:,2]) <= 30
    change = pred[1:] != pred[:-1]
    direct = arcs[pred[:-1],1] == arcs[pred[1:],0]
    ref_break = arcs[route[:-1],1] != arcs[route[1:],0]
    ref_tokens = route[np.r_[True, (route[1:] != route[:-1]) | ref_break]].tolist()
    pred_tokens = pred[np.r_[True, change | ~ok_step]].tolist()
    lcs = lcs_length(pred_tokens, ref_tokens)
    reverse = (arcs[h] == arcs[y][:,::-1]).all(axis=1) & (h != y)
    return dict(gps=len(track), eligible=len(y), candidates=int((ids >= 0).sum()),
                fallbacks=int(fallback.sum()), correct=int((y==h).sum()), reverse_errors=int(reverse.sum()),
                hits_at_2=int(matched[:,:2].any(axis=1).sum()),
                hits_at_4=int(matched[:,:4].any(axis=1).sum()),hits_at_8=int(matched.any(axis=1).sum()),
                macro_f1=float(np.mean(2*tp/(actual+guessed))) if len(y) else None,
                steps=int(ok_step.sum()), continuous=int((ok_step & (~change | direct)).sum()),
                changes=int((ok_step & change).sum()), connected_changes=int((ok_step & change & direct).sum()),
                predicted_tokens=len(pred_tokens), reference_tokens=len(ref_tokens), lcs=lcs,
                sequence_f1=2*lcs/(len(pred_tokens)+len(ref_tokens)))
