"""Fixed input interventions; no labels, raw clocks or fitted parameters."""
import hashlib

import numpy as np

NAMES = ('identity_anchor', 'fit_mean_content', 'mask_pattern_time_shuffle42')

def transform(image, name, sample_id):
    assert image.shape == (8, 30, 256) and image.dtype == np.float32
    assert name in NAMES
    result = image.copy()
    eligible = moved = 0
    if name == 'fit_mean_content':
        result[:3] = 0  # Zero in fit-standardized log1p space, NOT raw amplitude zero.
    elif name == 'mask_pattern_time_shuffle42':
        mask = image[3:6].transpose(2, 0, 1).astype(bool)
        pattern = np.packbits(mask.reshape(256, 90), axis=1)
        _, inverse = np.unique(pattern, axis=0, return_inverse=True)
        seed = int.from_bytes(hashlib.sha256(('42:'+sample_id).encode()).digest()[:8], 'little')
        rng = np.random.default_rng(seed)
        values = image[:3].transpose(2, 0, 1)
        shuffled = values.copy()
        for group in np.unique(inverse):
            indices = np.flatnonzero(inverse == group)
            # Entirely invisible frames carry no content and are never counted movable.
            if len(indices) < 2 or not mask[indices[0]].any():
                continue
            permutation = rng.permutation(indices)
            shuffled[indices] = values[permutation]
            eligible += len(indices); moved += int(np.count_nonzero(indices != permutation))
        result[:3] = shuffled.transpose(1, 2, 0)
    assert np.array_equal(result[3:], image[3:])
    valid = image[3:6].astype(bool)
    assert np.all(result[:3][~valid] == 0) and np.isfinite(result).all()
    return result, {'eligible_visible_frames': eligible, 'permuted_visible_frames': moved,
        'changed_visible_values': int(np.count_nonzero(result[:3][valid] != image[:3][valid])),
        'visible_values': int(valid.sum())}
