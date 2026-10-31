"""Gap-calibrated observed motion, antenna sets and event-time paired SSM.

Same C1 input. No raw filling, hidden values, extra fit statistics or pretrained
weights. This is a research revision, not a claim of a new set/SSM primitive.
"""

import math

import torch
from .state import PairedStateBlock
from torch import nn
from torch.nn import functional as F

class GapAwareMotion(nn.Module):
    def __init__(self, filters=6, lags=(1, 4, 16), max_gap_ratio=4):
        super().__init__()
        self.filters, self.lags, self.max_gap_ratio = filters, lags, max_gap_ratio
        self.basis = nn.Sequential(nn.Linear(9, 32), nn.Tanh(), nn.Linear(32, filters))

    def forward(self, image, timestamp):
        mask = image[:, 3:6] > 0.5
        x = torch.where(mask, image[:, :3], torch.zeros_like(image[:, :3]))
        b, a, f, t = x.shape
        ordinal = torch.arange(t, device=x.device).view(1, 1, 1, t)
        # Most recent ACTUALLY observed cell at/before each original grid index.
        latest = torch.where(mask, ordinal, -1).cummax(dim=-1).values
        clock = timestamp[:, None, None, :].expand(b, a, f, t)
        coordinate = image[:, 7, :, 0]
        terms = [coordinate]
        for k in range(1, 5):
            terms += [torch.sin(math.pi * k * coordinate), torch.cos(math.pi * k * coordinate)]
        weights = self.basis(torch.stack(terms, -1)).softmax(dim=1)
        features, masses = [], []
        for lag in self.lags:
            anchor = F.pad(latest[..., :-lag], (lag, 0), value=-1)
            previous = x.gather(-1, anchor.clamp_min(0))
            elapsed = clock - clock.gather(-1, anchor.clamp_min(0))
            # The intended lag remains on the original timestamp axis; do not
            # compress missing steps or pretend a distant pair is a short lag.
            desired = timestamp - timestamp[:, (torch.arange(t, device=x.device) - lag).clamp_min(0)]
            horizon = desired[:, None, None, :]
            support = (
                mask & (anchor >= 0) & (elapsed > 0)
                & (elapsed <= self.max_gap_ratio * horizon + 1e-7)
            ).to(x.dtype)
            change = (x - previous) * (horizon / elapsed.clamp_min(1e-6)) * support
            mass = torch.einsum("baft,bfk->bakt", support, weights)
            signed = torch.einsum("baft,bfk->bakt", change, weights) / mass.clamp_min(1e-8)
            energy = torch.einsum("baft,bfk->bakt", change.square(), weights) / mass.clamp_min(1e-8)
            rms = torch.sqrt(energy + 1e-8) - 1e-4
            features += [signed.permute(0, 1, 3, 2), rms.permute(0, 1, 3, 2)]
            masses.append(mass)
        # Per-antenna support remains separate until the set reduction.
        q = torch.stack(masses).mean(dim=(0, 3))
        return torch.cat(features, -1), q

def event_intervals(q, timestamp):
    t = q.shape[1]
    index = torch.arange(t, device=q.device)[None].expand_as(q)
    latest = torch.where(q > 0, index, -1).cummax(-1).values
    previous = F.pad(latest[:, :-1], (1, 0), value=-1)
    base = (timestamp[:, 1:2] - timestamp[:, :1]).clamp_min(1e-6)
    elapsed = timestamp - timestamp.gather(1, previous.clamp_min(0))
    elapsed = torch.where(previous >= 0, elapsed, base)
    return torch.where(q > 0, elapsed, 0), base

class EventStateBlock(PairedStateBlock):
    def forward(self, tokens, q, elapsed, base):
        z = self.norm(tokens)
        value = self.value(z).transpose(1, 2)
        delta = ((0.2 + 0.8 * self.delta(z).sigmoid()) * elapsed[:, :, None]).transpose(1, 2)
        # Elapsed time controls decay, while each observed token contributes one
        # original grid unit of evidence. A long gap must not create extra mass.
        evidence = q * base / elapsed.clamp_min(1e-6)
        mixed = self.integrate(value, evidence, delta)
        z = tokens + self.output(mixed)
        return (z + self.ff(z)) * (q > 0)[:, :, None]

class CSMISSM(nn.Module):
    def __init__(self, classes, width=64):
        super().__init__()
        self.motion = GapAwareMotion()
        self.antenna_encode = nn.Sequential(nn.Linear(36, width), nn.GELU(), nn.LayerNorm(width))
        self.set_encode = nn.Sequential(nn.Linear(2 * width, width), nn.GELU(), nn.LayerNorm(width))
        self.blocks = nn.ModuleList([EventStateBlock(width), EventStateBlock(width)])
        self.classifier = nn.Sequential(nn.LayerNorm(2 * width), nn.Linear(2 * width, classes))

    def tokens(self, image, timestamp):
        motion, support = self.motion(image, timestamp)
        encoded = self.antenna_encode(motion)
        count = support.sum(1).clamp_min(1e-8)
        mean = (encoded * support[..., None]).sum(1) / count[..., None]
        variance = ((encoded - mean[:, None]).square() * support[..., None]).sum(1) / count[..., None]
        deviation = torch.sqrt(variance + 1e-8) - 1e-4
        token = self.set_encode(torch.cat([mean, deviation], -1))
        antennas = (support > 0).sum(1).clamp_min(1)
        q = support.sum(1) / antennas
        return token * (q > 0)[..., None], q

    def representation(self, image, timestamp):
        if image.ndim != 4 or image.shape[1] != 8 or image.shape[-1] <= 16:
            raise ValueError("Expected common8/native-frequency/T>16 input")
        tokens, q = self.tokens(image, timestamp)
        elapsed, base = event_intervals(q, timestamp)
        for block in self.blocks:
            tokens = block(tokens, q, elapsed, base)
        count = q.sum(1, keepdim=True).clamp_min(1e-8)
        mean = (tokens * q[..., None]).sum(1) / count
        variance = ((tokens - mean[:, None]).square() * q[..., None]).sum(1) / count
        deviation = torch.sqrt(variance + 1e-8) - 1e-4
        return torch.cat([mean, deviation], -1) * (q.sum(1) > 0)[:, None]

    def forward(self, image, timestamp):
        return self.classifier(self.representation(image, timestamp))
