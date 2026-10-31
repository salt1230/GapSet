"""Three capacity-matched leave-one-component-out Widar controls.

All parameters/initialization order match frozen v2. Flags alter only the
declared computation. These are fresh training controls, not pretrained models.
"""

import math

import torch
from .state import PairedStateBlock
from torch import nn
from torch.nn import functional as F
from .v2 import EventStateBlock, GapAwareMotion, event_intervals

VARIANTS = ('fixed_lag', 'ordered_antennas', 'grid_state')

class ControlledMotion(GapAwareMotion):
    def __init__(self, fixed=False):
        super().__init__()
        self.fixed = fixed

    def forward(self, image, timestamp):
        if not self.fixed:
            return super().forward(image, timestamp)
        mask = image[:, 3:6] > .5
        x = torch.where(mask, image[:, :3], torch.zeros_like(image[:, :3]))
        coordinate = image[:, 7, :, 0]
        terms = [coordinate]
        for k in range(1, 5):
            terms += [torch.sin(math.pi * k * coordinate), torch.cos(math.pi * k * coordinate)]
        weights = self.basis(torch.stack(terms, -1)).softmax(dim=1)
        features, masses = [], []
        for lag in self.lags:
            past_x = F.pad(x[..., :-lag], (lag, 0))
            past_m = F.pad(mask[..., :-lag], (lag, 0))
            support = (mask & past_m).to(x.dtype)
            change = (x - past_x) * support
            mass = torch.einsum('baft,bfk->bakt', support, weights)
            signed = torch.einsum('baft,bfk->bakt', change, weights) / mass.clamp_min(1e-8)
            energy = torch.einsum('baft,bfk->bakt', change.square(), weights) / mass.clamp_min(1e-8)
            rms = torch.sqrt(energy + 1e-8) - 1e-4
            features += [signed.permute(0, 1, 3, 2), rms.permute(0, 1, 3, 2)]
            masses.append(mass)
        return torch.cat(features, -1), torch.stack(masses).mean(dim=(0, 3))

class ControlledState(EventStateBlock):
    def __init__(self, width, grid=False):
        super().__init__(width)
        self.grid = grid

    def forward(self, tokens, q, elapsed, base, grid_dt):
        if self.grid:
            return PairedStateBlock.forward(self, tokens, q, grid_dt)
        return super().forward(tokens, q, elapsed, base)

class ControlledCSMISSM(nn.Module):
    def __init__(self, variant, classes=6, width=64):
        super().__init__()
        if variant not in (*VARIANTS, 'full'):
            raise ValueError('Undeclared control')
        self.variant = variant
        # Exactly the same creation order/shapes as frozen v2.
        self.motion = ControlledMotion(fixed=variant == 'fixed_lag')
        self.antenna_encode = nn.Sequential(nn.Linear(36, width), nn.GELU(), nn.LayerNorm(width))
        self.set_encode = nn.Sequential(nn.Linear(2 * width, width), nn.GELU(), nn.LayerNorm(width))
        self.blocks = nn.ModuleList([ControlledState(width, variant == 'grid_state'), ControlledState(width, variant == 'grid_state')])
        self.classifier = nn.Sequential(nn.LayerNorm(2 * width), nn.Linear(2 * width, classes))

    def tokens(self, image, timestamp):
        motion, support = self.motion(image, timestamp)
        encoded = self.antenna_encode(motion)
        if self.variant == 'ordered_antennas':
            # Fixed positional antenna readout, with the SAME128 inputs and
            # trainable parameter count as set mean/std. No hidden antenna is
            # filled; a missing slot has zero contribution and is not reweighted.
            first = encoded.new_tensor([1., 2., 3.]).view(1, 3, 1, 1) / 6
            second = encoded.new_tensor([3., 2., 1.]).view(1, 3, 1, 1) / 6
            weighted = encoded * support[..., None]
            summary = torch.cat([(weighted * first).sum(1), (weighted * second).sum(1)], -1)
        else:
            count = support.sum(1).clamp_min(1e-8)
            mean = (encoded * support[..., None]).sum(1) / count[..., None]
            variance = ((encoded - mean[:, None]).square() * support[..., None]).sum(1) / count[..., None]
            deviation = torch.sqrt(variance + 1e-8) - 1e-4
            summary = torch.cat([mean, deviation], -1)
        token = self.set_encode(summary)
        antennas = (support > 0).sum(1).clamp_min(1)
        q = support.sum(1) / antennas
        return token * (q > 0)[..., None], q

    def representation(self, image, timestamp):
        if image.ndim != 4 or image.shape[1] != 8 or image.shape[-1] <= 16:
            raise ValueError('Expected common8/native-frequency/T>16 input')
        tokens, q = self.tokens(image, timestamp)
        elapsed, base = event_intervals(q, timestamp)
        grid_dt = torch.cat([timestamp[:, 1:2] - timestamp[:, :1], timestamp[:, 1:] - timestamp[:, :-1]], 1).clamp_min(1e-6)
        for block in self.blocks:
            tokens = block(tokens, q, elapsed, base, grid_dt)
        count = q.sum(1, keepdim=True).clamp_min(1e-8)
        mean = (tokens * q[..., None]).sum(1) / count
        variance = ((tokens - mean[:, None]).square() * q[..., None]).sum(1) / count
        deviation = torch.sqrt(variance + 1e-8) - 1e-4
        return torch.cat([mean, deviation], -1) * (q.sum(1) > 0)[:, None]

    def forward(self, image, timestamp):
        return self.classifier(self.representation(image, timestamp))
