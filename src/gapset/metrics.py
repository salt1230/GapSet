"""Six-class primary metrics; undefined class F1 is zero."""
import numpy as np

def metrics(labels, predictions, classes):
    labels = np.asarray(labels)
    predictions = np.asarray(predictions)
    if labels.ndim != 1 or predictions.shape != labels.shape or not len(labels):
        raise ValueError("Bad prediction shape")
    if not np.issubdtype(labels.dtype, np.integer) or not np.issubdtype(
        predictions.dtype, np.integer
    ):
        raise ValueError("Integer labels required")
    if (
        np.any(labels < 0)
        or np.any(labels >= classes)
        or np.any(predictions < 0)
        or np.any(predictions >= classes)
    ):
        raise ValueError("Unknown class")
    matrix = np.bincount(labels * classes + predictions, minlength=classes * classes).reshape(
        classes, classes
    )
    true = np.diag(matrix)
    denominator = matrix.sum(0) + matrix.sum(1)
    f1 = np.divide(2 * true, denominator, out=np.zeros(classes, dtype=float), where=denominator > 0)
    return {
        "n": len(labels),
        "accuracy_percent": float(true.sum() / len(labels) * 100),
        "macro_f1_percent": float(f1.mean() * 100),
    }
