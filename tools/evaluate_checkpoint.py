"""Explicit evaluation of released paper weights; never trains or selects a model."""
import argparse
import json
import os
from pathlib import Path
import time


class IntervenedDataset:
    def __init__(self, base, intervention):
        self.base, self.intervention = base, intervention

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        import torch
        from gapset.analysis.interventions import transform
        image, timestamp, label, index = self.base[i]
        image, _ = transform(image.numpy(), self.intervention, self.base.rows[i]['id'])
        return torch.from_numpy(image), timestamp, label, index


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seed', type=int, choices=[42, 43, 44], required=True)
    p.add_argument('--checkpoints', type=Path, default=Path('checkpoints'))
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--input-identity', type=Path, required=True)
    p.add_argument('--expected-masks', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--split', choices=['both', 'source_val_clean', 'cross_domain_dev'], default='both')
    p.add_argument('--conditions', choices=['all', 'clean'], default='all')
    p.add_argument('--intervention', choices=['identity_anchor', 'fit_mean_content',
                   'mask_pattern_time_shuffle42'], default='identity_anchor')
    p.add_argument('--max-hours', type=float, default=12)
    p.add_argument('--lock', type=Path, default=Path('outputs/.gpu.lock'))
    a = p.parse_args()
    if a.output.exists():
        p.error('Use a new output directory; completed evaluations are never overwritten')
    if not 0 < a.max_hours <= 48:
        p.error('--max-hours must be in (0,48]')
    if a.intervention != 'identity_anchor' and (a.seed != 42 or a.conditions != 'clean'):
        p.error('Paper content interventions use fixed seed42 weights and clean input')
    import hashlib
    import numpy as np
    import torch
    from gapset.checkpoints import load_paper_model
    from gapset.cli import single_gpu_lock
    from gapset.data.dataset import C1Dataset, conditions
    from gapset.data.preprocess import digest, save_json
    from gapset.training.guard import Guard
    from gapset.training import worker
    worker.SEED = a.seed
    loader, run_inference = worker.loader, worker.run_inference
    identity = json.loads(a.input_identity.read_text(encoding='utf8'))
    frozen = json.loads(a.expected_masks.read_text(encoding='utf8'))
    manifest_hash = digest(a.data / 'manifest.json')
    if identity['status'] != 'passed' or identity['manifest_sha256'] != manifest_hash:
        raise ValueError('Audit does not match the supplied cache')
    if frozen['input_manifest_sha256'] != manifest_hash:
        raise ValueError('Frozen masks do not match the supplied cache')
    expected = {(x['split'], x['condition']): x['mask_sha256'] for x in frozen['tests']}
    model, entry = load_paper_model(a.seed, a.checkpoints)
    os.environ['GAPSET_RUN_DEADLINE'] = str(time.time() + a.max_hours * 3600)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    roles = ['source_val_clean', 'cross_domain_dev'] if a.split == 'both' else [a.split]
    with single_gpu_lock(a.lock):
        a.output.mkdir(parents=True)
        guard = Guard(a.output)
        guard.admit()
        model = model.cuda()
        results = []
        for role in roles:
            panel = conditions() if a.conditions == 'all' else [conditions()[0]]
            for condition in panel:
                ds = C1Dataset(a.data, role, condition)
                mask_hash = hashlib.sha256()
                for i, row in enumerate(ds.rows):
                    guard.check()
                    mask_hash.update(row['id'].encode())
                    mask_hash.update(ds.observation(i).valid.tobytes())
                if mask_hash.hexdigest() != expected[(role, condition['name'])]:
                    raise ValueError('Mask identity changed before evaluation')
                evaluated = ds if a.intervention == 'identity_anchor' else IntervenedDataset(ds, a.intervention)
                logits, labels, metrics = run_inference(model, loader(evaluated), guard)
                folder = a.output / role / condition['name']
                folder.mkdir(parents=True)
                predictions = logits.argmax(1)
                cm = np.bincount(labels * 6 + predictions, minlength=36).reshape(6, 6)
                files = {}
                for name, values in [('labels', labels), ('logits', logits), ('predictions', predictions)]:
                    path = folder / (name + '.npy')
                    np.save(path, values, allow_pickle=False)
                    files[name + '_file'] = str(path)
                    files[name + '_file_sha256'] = digest(path)
                row = dict(metrics, model='ours_gapset_relative_mass_control',
                           training_seed=a.seed, selected_epoch=entry['selected_epoch'],
                           split=role, condition=condition['name'], challenge=condition,
                           confusion=cm.tolist(), n=len(labels), status='complete',
                           mask_sha256=mask_hash.hexdigest(), input_manifest_sha256=manifest_hash,
                           selected_checkpoint_sha256=entry['release_sha256'],
                           original_checkpoint_sha256=entry['original_checkpoint_sha256'], **files)
                row['input_intervention'] = a.intervention
                save_json(folder / 'metrics.json', row)
                results.append(row)
                save_json(a.output / 'progress.json', {'completed': len(results), 'total': len(roles) * len(panel)})
        save_json(a.output / 'completion.json', {'status': 'complete', 'rows': results,
                  'scope': 'source/development evaluation', 'target_test_accessed': False})


if __name__ == '__main__':
    main()
