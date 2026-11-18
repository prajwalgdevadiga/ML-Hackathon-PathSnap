import numpy as np

FLOOR=1e-12

def predict(model, cols, features, ids):
    """One distribution per observation. The function has no scoring inputs."""
    valid=ids>=0
    positive=np.clip(model.predict_proba(features[valid][:,cols])[:,1].astype(float),FLOOR,1-FLOOR)
    odds=np.zeros(ids.shape)
    odds[valid]=positive/(1-positive)
    probs=odds/odds.sum(axis=1,keepdims=True)
    assert np.isfinite(probs).all() and np.allclose(probs.sum(axis=1),1)
    order=np.argsort(-probs,axis=1,kind="stable")
    return np.take_along_axis(ids,order,axis=1),probs
