"""Load only the three checksummed GapSet-R paper checkpoints."""
import hashlib
import json
from pathlib import Path

import torch

from .models.relative_model import RelativeCSMISSM


def load_paper_model(seed=42, directory='checkpoints', device='cpu'):
    """Return (eval-mode model, metadata); no forward or GPU allocation by default.

    Inputs to the model must follow the common8 fit-normalized contract. These
    weights do not contain a raw-CSI preprocessor or an independent test result.
    """
    if seed not in (42, 43, 44):
        raise ValueError('The paper uses training seeds 42, 43 and 44')
    directory = Path(directory)
    manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf8'))
    entry = next(x for x in manifest['checkpoints'] if x['seed'] == seed)
    if entry['status'] != 'included' or not entry['release_sha256']:
        raise FileNotFoundError(f'Paper checkpoint s{seed} has not been included')
    path = directory / entry['file']
    if hashlib.sha256(path.read_bytes()).hexdigest() != entry['release_sha256']:
        raise ValueError(f'Checkpoint checksum mismatch: {path.name}')
    state = torch.load(path, map_location='cpu', weights_only=True)
    if state['epoch'] != entry['selected_epoch'] or state['recipe']['seed'] != seed:
        raise ValueError('Checkpoint seed/epoch does not match its paper record')
    if state['recipe']['model'] != manifest['model']:
        raise ValueError('Only the final GapSet-R model is supported')
    model = RelativeCSMISSM()
    model.load_state_dict(state['model'], strict=True)
    if sum(x.numel() for x in model.parameters()) != 73100:
        raise ValueError('Unexpected model architecture')
    return model.to(device).eval(), dict(entry)
