"""Reproduce Table6's fit-only endpoint/mask ridge classifiers from a user's cache."""
import argparse
import json
from pathlib import Path

import numpy as np

from gapset.analysis.cue_features import mask_features, metrics
from gapset.data.preprocess import digest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        p.error('Choose a new output directory')
    manifest_hash = digest(a.data / 'manifest.json')
    manifest = json.loads((a.data / 'manifest.json').read_text(encoding='utf8'))
    if manifest['status'] != 'ready':
        raise ValueError('A verified cache is required')
    rows = manifest['records']
    valid = np.memmap(a.data / 'valid.bin', mode='r', dtype=bool, shape=tuple(manifest['shape']))
    features = np.empty((len(rows), 861))
    lengths = np.empty(len(rows))
    for i, row in enumerate(rows):
        lengths[i], features[i] = mask_features(valid[row['row']])
    labels = np.array([r['label'] for r in rows])
    roles = np.array([r['role'] for r in rows])
    fit = roles == 'fit_clean'
    centers = np.quantile(lengths[fit], np.linspace(0, 1, 12))
    width = max(float(lengths[fit].std()), 1.)
    endpoint = np.exp(-.5*((lengths[:, None]-centers)/width)**2)
    result = []
    for name, values in [('visible_endpoint_only', endpoint), ('mask_marginals_only', features)]:
        mu = values[fit].mean(0)
        sd = np.maximum(values[fit].std(0), 1e-8)
        design = np.column_stack([(values-mu)/sd, np.ones(len(rows))])
        regularizer = np.eye(design.shape[1])
        regularizer[-1, -1] = 0
        gram = design[fit].T @ design[fit] + regularizer
        weight = np.linalg.solve(gram, design[fit].T @ np.eye(6)[labels[fit]])
        scores = (design @ weight).astype(np.float32)
        for role in ['fit_clean', 'source_val_clean', 'cross_domain_dev']:
            take = roles == role
            result.append({'model': name, 'role': role, **metrics(labels[take], scores[take].argmax(1))})
    if digest(a.data / 'manifest.json') != manifest_hash:
        raise ValueError('Cache manifest changed')
    a.output.mkdir(parents=True)
    (a.output / 'summary.json').write_text(json.dumps({'rows': result, 'ridge_lambda': 1.,
           'manifest_sha256': manifest_hash, 'normalization_fitted_role': 'fit_clean',
           'amplitude_used': False, 'scope': 'window-level scores, not causal attribution'}, indent=2), encoding='utf8')


if __name__ == '__main__':
    main()
