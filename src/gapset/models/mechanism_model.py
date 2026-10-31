"""Two fixed support-normalization controls, matched to the original full v2."""
import torch
from .state import reference_scan
from torch import nn
from .v2 import EventStateBlock, GapAwareMotion, event_intervals

VARIANTS = ('state_no_mass_norm', 'ordered_normalized')

def ordered_summary(encoded, support):
    weighted = encoded * support[..., None]
    summaries = []
    for positions in ([1., 2., 3.], [3., 2., 1.]):
        weights = encoded.new_tensor(positions).view(1, 3, 1, 1) / 6
        mass = (support[..., None] * weights).sum(1)
        summaries.append((weighted * weights).sum(1) / mass.clamp_min(1e-8))
    return torch.cat(summaries, -1)

class MechanismStateBlock(EventStateBlock):
    def __init__(self, width, no_mass_norm=False):
        super().__init__(width)
        self.no_mass_norm = no_mass_norm

    def integrate(self, value, q, delta):
        if not self.no_mass_norm:
            return super().integrate(value, q, delta)
        # Keep the same paired scan, parameters, support injection and zero-mass
        # guard. The ONLY altered operation is removal of the mass division.
        u = torch.cat([value*q[:, None], q[:, None].expand_as(value)], 1)
        paired_delta = torch.cat([delta, delta], 1).contiguous()
        a = -self.log_rates.exp()
        c = torch.nn.functional.softplus(self.readout)
        paired_a, paired_c = torch.cat([a, a]), torch.cat([c, c])
        if u.is_cuda and not self.force_reference:
            from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
            both = selective_scan_fn(u.contiguous(), paired_delta, paired_a,
                torch.ones_like(paired_a), paired_c)
        else:
            both = reference_scan(u, paired_delta, paired_a, paired_c)
        numerator, mass = both.chunk(2, 1)
        return torch.where(mass > 1e-8, numerator, 0).transpose(1, 2)

class MechanismCSMISSM(nn.Module):
    def __init__(self, variant, classes=6, width=64):
        super().__init__()
        if variant not in (*VARIANTS, 'full'):
            raise ValueError('Undeclared support mechanism control')
        self.variant = variant
        # Preserve full-v2 parameter names, shapes and random construction order.
        self.motion = GapAwareMotion()
        self.antenna_encode = nn.Sequential(nn.Linear(36, width), nn.GELU(), nn.LayerNorm(width))
        self.set_encode = nn.Sequential(nn.Linear(2*width, width), nn.GELU(), nn.LayerNorm(width))
        self.blocks = nn.ModuleList([MechanismStateBlock(width, variant == 'state_no_mass_norm'),
                                    MechanismStateBlock(width, variant == 'state_no_mass_norm')])
        self.classifier = nn.Sequential(nn.LayerNorm(2*width), nn.Linear(2*width, classes))

    def tokens(self, image, timestamp):
        motion, support = self.motion(image, timestamp)
        encoded = self.antenna_encode(motion)
        if self.variant == 'ordered_normalized':
            summary = ordered_summary(encoded, support)
        else:
            count = support.sum(1).clamp_min(1e-8)
            mean = (encoded*support[..., None]).sum(1)/count[..., None]
            variance = ((encoded-mean[:, None]).square()*support[..., None]).sum(1)/count[..., None]
            deviation = torch.sqrt(variance+1e-8)-1e-4
            summary = torch.cat([mean, deviation], -1)
        token = self.set_encode(summary)
        antennas = (support > 0).sum(1).clamp_min(1)
        q = support.sum(1)/antennas
        return token*(q > 0)[..., None], q

    def representation(self, image, timestamp):
        if image.ndim != 4 or image.shape[1] != 8 or image.shape[-1] <= 16:
            raise ValueError('Expected common8/native-frequency/T>16 input')
        tokens, q = self.tokens(image, timestamp)
        elapsed, base = event_intervals(q, timestamp)
        for block in self.blocks:
            tokens = block(tokens, q, elapsed, base)
        count = q.sum(1, keepdim=True).clamp_min(1e-8)
        mean = (tokens*q[..., None]).sum(1)/count
        variance = ((tokens-mean[:, None]).square()*q[..., None]).sum(1)/count
        deviation = torch.sqrt(variance+1e-8)-1e-4
        return torch.cat([mean, deviation], -1)*(q.sum(1) > 0)[:, None]

    def forward(self, image, timestamp):
        return self.classifier(self.representation(image, timestamp))
