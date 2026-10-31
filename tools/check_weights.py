"""Check released weight hashes and strict loading, without model inference."""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory', type=Path, default=Path('checkpoints'))
    a = p.parse_args()
    import torch
    from gapset.checkpoints import load_paper_model
    records = []
    for seed in (42, 43, 44):
        model, entry = load_paper_model(seed, a.directory)
        records.append({'seed': seed, 'epoch': entry['selected_epoch'],
                        'parameters': sum(x.numel() for x in model.parameters()),
                        'sha256': entry['release_sha256'], 'strict_load': True})
    if torch.cuda.is_initialized():
        raise RuntimeError('Weight verification must not initialize CUDA')
    print(json.dumps({'checkpoints': records, 'model_forward_executed': False}, indent=2))


if __name__ == '__main__':
    main()
