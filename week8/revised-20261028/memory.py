"""Key-value memory stores and recalled picture experiment."""
import numpy as np


def recall(keys, values, index, method, beta=8):
    keys = np.asarray(keys, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    if keys.ndim != 2 or values.ndim != 2 or len(keys) != len(values):
        raise ValueError('keys and values must be two-dimensional with matching rows')
    if not 0 <= index < len(keys):
        raise ValueError('query index is out of range')
    scores = keys @ keys[index]
    if method == 'outer':
        weights = scores
    elif method == 'relu':
        weights = np.maximum(scores, 0)
    elif method == 'relu2':
        weights = np.maximum(scores, 0)**2
    elif method == 'softmax':
        logits = beta * scores
        weights = np.exp(logits-np.max(logits))
        weights /= weights.sum()
    else:
        raise ValueError(f'unknown store: {method}')
    own_weight = weights[index]
    if own_weight <= 0:
        raise ValueError('query has nonpositive self-weight')
    signal = own_weight * values[index]
    interference = weights @ values - signal
    denominator = np.linalg.norm(signal)
    if denominator == 0:
        raise ValueError('stored value has zero norm')
    return (signal + interference) / own_weight, float(np.linalg.norm(interference) / denominator)
