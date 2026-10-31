"""Fixed submission evidence models. No score-dependent architecture choices.

GRU-D equations follow Che et al. (2018), equations 4--6 and GRU gates.
This is an explicitly disclosed PyTorch/common8 adaptation, not author code.
The full grid is processed and read at the last publicly visible CSI frame.
"""
import json
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

@torch.jit.script
def recurrent_scan(projected: torch.Tensor, decay: torch.Tensor,
                   recurrent: torch.Tensor, last: torch.Tensor):
    width = recurrent.shape[1]
    state = projected.new_zeros((projected.shape[1], width))
    selected = torch.zeros_like(state)
    for t in range(projected.shape[0]):
        previous = decay[t] * state
        gates = torch.sigmoid(projected[t, :, :2*width] + F.linear(previous, recurrent[:2*width]))
        reset, update = gates.chunk(2, dim=-1)
        candidate = torch.tanh(projected[t, :, 2*width:] + F.linear(reset*previous, recurrent[2*width:]))
        state = (1-update)*previous + update*candidate
        selected = torch.where((last == t)[:, None], state, selected)
    return selected

class GRUDCommon8(nn.Module):
    def __init__(self, fit_mean, width=64, classes=6):
        super().__init__()
        means = torch.as_tensor(fit_mean, dtype=torch.float32)
        if means.shape != (90,) or not torch.isfinite(means).all():
            raise ValueError('Expected fit-only means for all native CSI variables')
        self.register_buffer('fit_mean', means)
        self.decay_x_weight = nn.Parameter(torch.empty(90))
        self.decay_x_bias = nn.Parameter(torch.empty(90))
        self.decay_hidden = nn.Linear(240, width)
        self.input_linear = nn.Linear(240, 3*width)
        self.mask_linear = nn.Linear(240, 3*width, bias=False)
        self.recurrent = nn.Parameter(torch.empty(3*width, width))
        self.classifier = nn.Linear(width, classes)
        bound = 1/math.sqrt(width)
        nn.init.uniform_(self.decay_x_weight, -bound, bound)
        nn.init.uniform_(self.decay_x_bias, -bound, bound)
        nn.init.uniform_(self.recurrent, -bound, bound)

    def prepare(self, image, timestamp):
        if image.shape[1:] != (8, 30, 256):
            raise ValueError('GRU-D adaptation is fixed to the common8 native grid')
        batch, _, _, steps = image.shape
        m = (image[:, 3:6] > .5).reshape(batch, 90, steps)
        raw = torch.where(m, image[:, :3].reshape(batch, 90, steps), 0)
        index = torch.arange(steps, device=image.device)[None, None]
        latest = torch.where(m, index, -1).cummax(-1).values
        previous = F.pad(latest[..., :-1], (1, 0), value=-1)
        clock = timestamp[:, None].expand(batch, 90, steps)
        since = clock - clock.gather(-1, previous.clamp_min(0))
        # Before first observation, paper recurrence accumulates from time zero.
        since = torch.where(previous >= 0, since, clock-timestamp[:, :1, None])
        last_value = raw.gather(-1, previous.clamp_min(0))
        mean = self.fit_mean[None, :, None]
        last_value = torch.where(previous >= 0, last_value, mean)
        gamma = torch.exp(-F.relu(self.decay_x_weight[None, :, None]*since + self.decay_x_bias[None, :, None]))
        filled = torch.where(m, raw, gamma*last_value+(1-gamma)*mean)
        auxiliary = image[:, 3:].reshape(batch, 150, steps)
        values = torch.cat([filled, auxiliary], 1).transpose(1, 2)
        masks = torch.cat([m.to(image.dtype), torch.ones_like(auxiliary)], 1).transpose(1, 2)
        grid_delta = torch.cat([timestamp[:, :1]*0, timestamp[:, 1:]-timestamp[:, :-1]], 1)
        intervals = torch.cat([since, grid_delta[:, None].expand(batch, 150, steps)], 1).transpose(1, 2)
        last = torch.where(m.any(1), index[:, 0], -1).amax(1)
        return values, masks, intervals, last

    def forward(self, image, timestamp):
        values, masks, intervals, last = self.prepare(image, timestamp)
        projected = self.input_linear(values)+self.mask_linear(masks)
        decay = torch.exp(-F.relu(self.decay_hidden(intervals)))
        state = recurrent_scan(projected.transpose(0, 1), decay.transpose(0, 1), self.recurrent, last)
        return self.classifier(state)

def make_model(name, vendor):
    if name == 'grud_common8':
        record = json.loads((Path(vendor)/'fit_means.json').read_text())
        assert record['role'] == 'fit_clean' and record['no_validation_or_development_used']
        return GRUDCommon8(record['means_normalized_csi'])
    if name == 'ours_gapset_relative_fixed_lag':
        # All parameter names, shapes and construction order match final GapSet-R.
        from .relative_model import RelativeCSMISSM, RelativeStateBlock
        from .control_reference import ControlledMotion

        class FixedLagRelative(RelativeCSMISSM):
            def __init__(self):
                nn.Module.__init__(self)
                self.motion = ControlledMotion(fixed=True)
                self.antenna_encode = nn.Sequential(nn.Linear(36, 64), nn.GELU(), nn.LayerNorm(64))
                self.set_encode = nn.Sequential(nn.Linear(128, 64), nn.GELU(), nn.LayerNorm(64))
                self.blocks = nn.ModuleList([RelativeStateBlock(64), RelativeStateBlock(64)])
                self.classifier = nn.Sequential(nn.LayerNorm(128), nn.Linear(128, 6))
        return FixedLagRelative()
    raise ValueError(name)

NAMES = ('grud_common8', 'ours_gapset_relative_fixed_lag')
YEARS = {NAMES[0]: 2018, NAMES[1]: 2026}

def core_paths(name):
    if name == NAMES[0]:
        return ('input_linear', 'mask_linear', 'decay_hidden', 'classifier')
    return ('motion', 'antenna_encode', 'set_encode', 'blocks.0', 'blocks.1', 'classifier')
