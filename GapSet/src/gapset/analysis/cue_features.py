"""Exact endpoint and 861 mask-marginal feature definitions."""
import numpy as np

def mask_features(mask):
    """One visible endpoint and861 marginals; never uses amplitude/ticks/labels."""
    assert mask.shape == (256, 3, 30) and mask.dtype == bool
    visible = mask.any((1, 2)); end = np.flatnonzero(visible)
    length = int(end[-1]+1) if len(end) else 0
    summary = np.concatenate([mask.mean(2).ravel(), mask.mean(0).ravel(), mask.mean((0, 2))])
    assert summary.shape == (861,)
    return length, summary

def metrics(y, pred):
    cm = np.bincount(y*6+pred, minlength=36).reshape(6, 6)
    denominator = cm.sum(0)+cm.sum(1)
    f1 = np.divide(2*cm.diagonal(), denominator, out=np.zeros(6), where=denominator > 0)
    return {'n': len(y), 'accuracy_percent': 100*cm.trace()/len(y),
            'macro_f1_percent': 100*f1.mean(), 'confusion': cm.tolist()}
