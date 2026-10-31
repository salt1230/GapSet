"""Common-support motion features and paired evidence-normalized selective SSM.

This is a research candidate, not standard Mamba or a calibrated RF filter.
CUDA uses the existing Mamba selective-scan primitive; CPU has a literal
recurrence for independent numerical checks. No extra data or normalization fit.
"""

import math

import torch
from torch import nn

def reference_scan(u, delta, a, c):
    state = u.new_zeros(u.shape[0], u.shape[1], a.shape[1])
    values = []
    for t in range(u.shape[-1]):
        dt = delta[:, :, t, None]
        state = torch.exp(dt * a[None]) * state + dt * u[:, :, t, None]
        values.append((state * c[None]).sum(-1))
    return torch.stack(values, -1)

class CommonSupportMotion(nn.Module):
    def __init__(self, filters=6, lags=(1, 4, 16)):
        super().__init__()
        self.filters = filters
        self.lags = lags
        self.basis = nn.Sequential(nn.Linear(9, 32), nn.Tanh(), nn.Linear(32, filters))

    def forward(self, image):
        mask = image[:, 3:6] > 0.5
        x = torch.where(mask, image[:, :3], torch.zeros_like(image[:, :3]))
        # Published ordinal coordinate, not an inferred RF frequency/bandwidth.
        coordinate = image[:, 7, :, 0]
        terms = [coordinate]
        for k in range(1, 5):
            terms.extend([torch.sin(math.pi * k * coordinate), torch.cos(math.pi * k * coordinate)])
        weights = self.basis(torch.stack(terms, -1)).softmax(dim=1)  # B,F,K positive
        features, masses = [], []
        for lag in self.lags:
            past_x = torch.nn.functional.pad(x[..., :-lag], (lag, 0))
            past_m = torch.nn.functional.pad(mask[..., :-lag], (lag, 0))
            support = (mask & past_m).to(x.dtype)
            change = (x - past_x) * support
            mass = torch.einsum("baft,bfk->bakt", support, weights)
            signed = torch.einsum("baft,bfk->bakt", change, weights)
            energy = torch.einsum("baft,bfk->bakt", change.square(), weights)
            denominator = mass.clamp_min(1e-8)
            signed = signed / denominator
            rms = torch.sqrt(energy / denominator + 1e-8) - 1e-4
            features.extend([signed.flatten(1, 2), rms.flatten(1, 2)])
            masses.append(mass)
        q = torch.stack(masses).mean(dim=(0, 2, 3))  # B,T; actual support, not quality proxy
        return torch.cat(features, dim=1).transpose(1, 2), q

class PairedStateBlock(nn.Module):
    def __init__(self, width=64, states=8):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.value = nn.Linear(width, width)
        self.delta = nn.Linear(width, width)
        initial = torch.linspace(math.log(0.5), math.log(32), states)
        self.log_rates = nn.Parameter(initial.repeat(width, 1))
        self.readout = nn.Parameter(torch.zeros(width, states))
        self.output = nn.Linear(width, width)
        self.ff = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, width),
        )
        self.force_reference = False

    def integrate(self, value, q, delta):
        u = torch.cat([value * q[:, None], q[:, None].expand_as(value)], dim=1)
        paired_delta = torch.cat([delta, delta], dim=1).contiguous()
        a = -self.log_rates.exp()
        c = torch.nn.functional.softplus(self.readout)
        paired_a, paired_c = torch.cat([a, a]), torch.cat([c, c])
        if u.is_cuda and not self.force_reference:
            from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

            both = selective_scan_fn(
                u.contiguous(), paired_delta, paired_a, torch.ones_like(paired_a), paired_c
            )
        else:
            both = reference_scan(u, paired_delta, paired_a, paired_c)
        numerator, mass = both.chunk(2, dim=1)
        return torch.where(mass > 1e-8, numerator / mass.clamp_min(1e-8), 0).transpose(1, 2)

    def forward(self, tokens, q, dt):
        z = self.norm(tokens)
        value = self.value(z).transpose(1, 2)
        # Same content-dependent positive kernel for feature and support mass.
        delta = ((0.2 + 0.8 * self.delta(z).sigmoid()) * dt[:, :, None]).transpose(1, 2)
        mixed = self.integrate(value, q, delta)
        z = tokens + self.output(mixed)
        return (z + self.ff(z)) * (q > 0)[:, :, None]

class CSMISSM(nn.Module):
    def __init__(self, classes, width=64):
        super().__init__()
        self.motion = CommonSupportMotion()
        self.encode = nn.Sequential(nn.Linear(108, width), nn.GELU(), nn.LayerNorm(width))
        self.blocks = nn.ModuleList([PairedStateBlock(width), PairedStateBlock(width)])
        self.classifier = nn.Sequential(nn.LayerNorm(2 * width), nn.Linear(2 * width, classes))

    def representation(self, image, timestamp):
        if image.ndim != 4 or image.shape[1] != 8 or image.shape[-1] <= 16:
            raise ValueError("Expected common8/native-frequency/T>16 input")
        motion, q = self.motion(image)
        tokens = self.encode(motion) * (q > 0)[:, :, None]
        dt = torch.cat(
            [timestamp[:, 1:2] - timestamp[:, :1], timestamp[:, 1:] - timestamp[:, :-1]], dim=1
        ).clamp_min(1e-6)
        for block in self.blocks:
            tokens = block(tokens, q, dt)
        count = q.sum(1, keepdim=True).clamp_min(1e-8)
        mean = (tokens * q[:, :, None]).sum(1) / count
        variance = ((tokens - mean[:, None]).square() * q[:, :, None]).sum(1) / count
        deviation = torch.sqrt(variance + 1e-8) - 1e-4
        representation = torch.cat([mean, deviation], dim=1)
        return representation * (q.sum(1) > 0)[:, None]

    def forward(self, image, timestamp):
        return self.classifier(self.representation(image, timestamp))
