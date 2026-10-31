"""One fixed hypothesis: preserve dense-state scale while correcting support.

The reference is an algebraic full-grid support schedule, not imputed CSI or
counterfactual hidden content. Its gates use only current observed tokens.
"""
import torch
from .state import reference_scan
from torch import nn
from .v2 import CSMISSM, EventStateBlock, GapAwareMotion, event_intervals

def ideal_support(q, lags=(1, 4, 16)):
    index = torch.arange(q.shape[1], device=q.device)
    return torch.stack([index >= lag for lag in lags]).to(q.dtype).mean(0).expand_as(q)

class RelativeStateBlock(EventStateBlock):
    def integrate_relative(self, value, q, delta, reference_q, reference_delta):
        # Actual feature numerator and mass retain exactly the v2 equations.
        # The third scan measures full-grid exposure using the same learned
        # rates/readout and content gates, but grid rather than event intervals.
        u = torch.cat([value*q[:, None], q[:, None].expand_as(value),
                       reference_q[:, None].expand_as(value)], 1).contiguous()
        dt = torch.cat([delta, delta, reference_delta], 1).contiguous()
        a = -self.log_rates.exp()
        c = torch.nn.functional.softplus(self.readout)
        triple_a = torch.cat([a, a, a])
        triple_c = torch.cat([c, c, c])
        if u.is_cuda and not self.force_reference:
            from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
            both = selective_scan_fn(u, dt, triple_a, torch.ones_like(triple_a), triple_c)
        else:
            both = reference_scan(u, dt, triple_a, triple_c)
        numerator, mass, reference_mass = both.chunk(3, 1)
        output = torch.where(mass > 1e-8,
                             (numerator/mass.clamp_min(1e-8))*reference_mass, 0)
        return output.transpose(1, 2)

    def forward(self, tokens, q, elapsed, base, reference_q):
        z = self.norm(tokens)
        value = self.value(z).transpose(1, 2)
        gate = 0.2 + 0.8*self.delta(z).sigmoid()
        delta = (gate*elapsed[:, :, None]).transpose(1, 2)
        reference_delta = (gate*base[:, :, None]).transpose(1, 2)
        evidence = q*base/elapsed.clamp_min(1e-6)
        mixed = self.integrate_relative(value, evidence, delta, reference_q, reference_delta)
        z = tokens + self.output(mixed)
        return (z+self.ff(z))*(q > 0)[:, :, None]

class RelativeCSMISSM(CSMISSM):
    def __init__(self, classes=6, width=64):
        # Match parameter construction order; do not construct then replace old
        # blocks, which would consume a different initialization RNG stream.
        nn.Module.__init__(self)
        self.motion = GapAwareMotion()
        self.antenna_encode = nn.Sequential(nn.Linear(36, width), nn.GELU(), nn.LayerNorm(width))
        self.set_encode = nn.Sequential(nn.Linear(2*width, width), nn.GELU(), nn.LayerNorm(width))
        self.blocks = nn.ModuleList([RelativeStateBlock(width), RelativeStateBlock(width)])
        self.classifier = nn.Sequential(nn.LayerNorm(2*width), nn.Linear(2*width, classes))

    def representation(self, image, timestamp):
        if image.ndim != 4 or image.shape[1] != 8 or image.shape[-1] <= 16:
            raise ValueError('Expected common8/native-frequency/T>16 input')
        tokens, q = self.tokens(image, timestamp)
        elapsed, base = event_intervals(q, timestamp)
        reference_q = ideal_support(q, self.motion.lags)
        for block in self.blocks:
            tokens = block(tokens, q, elapsed, base, reference_q)
        count = q.sum(1, keepdim=True).clamp_min(1e-8)
        mean = (tokens*q[..., None]).sum(1)/count
        variance = ((tokens-mean[:, None]).square()*q[..., None]).sum(1)/count
        deviation = torch.sqrt(variance+1e-8)-1e-4
        return torch.cat([mean, deviation], -1)*(q.sum(1) > 0)[:, None]
