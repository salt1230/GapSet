"""Paper model constructors. No pretrained weights or target-domain adaptation."""
from pathlib import Path

BASELINES=('sensefi_mlp','sensefi_lenet','sensefi_resnet18','sensefi_lstm','sensefi_vit',
 'amcs_tcn_c1','dfs_csi_backbone','sensefi_cnn_gru','wiflexformer','csibert2_clean_backbone','hms_rwkv_full')
CONTROLS=('ours_gapset_relative_mass_control','ours_gapset_state_no_mass_norm_control',
 'ours_gapset_relative_fixed_lag','absolute_norm','early_fixed_lag','ordered_antennas',
 'ordered_normalized','grid_state')
NAMES=CONTROLS+('grud_common8',)+BASELINES
YEARS={name:2026 for name in NAMES}
YEARS.update(sensefi_mlp=2023,sensefi_lenet=1998,sensefi_resnet18=2016,
 sensefi_lstm=1997,sensefi_vit=2021,sensefi_cnn_gru=2023,wiflexformer=2024,grud_common8=2018)

def make_model(name, vendor):
    vendor=Path(vendor)
    if name=='ours_gapset_relative_mass_control':
        from .relative_model import RelativeCSMISSM
        return RelativeCSMISSM()
    if name in ('grud_common8','ours_gapset_relative_fixed_lag'):
        from .submission_models import make_model as build
        return build(name,vendor)
    if name in ('ours_gapset_state_no_mass_norm_control','ordered_normalized'):
        from .mechanism_model import MechanismCSMISSM
        return MechanismCSMISSM('state_no_mass_norm' if name.startswith('ours_') else name)
    if name=='absolute_norm':
        from .v2 import CSMISSM
        return CSMISSM()
    if name in ('early_fixed_lag','ordered_antennas','grid_state'):
        from .control_reference import ControlledCSMISSM
        return ControlledCSMISSM('fixed_lag' if name=='early_fixed_lag' else name)
    from . import baseline_original as original,baseline_expanded as expanded
    if name in original.NAMES:return original.make_model(name,vendor)
    if name in expanded.NAMES:return expanded.make_model(name,vendor)
    raise ValueError('Unknown declared model: '+name)

def core_paths(name):
    if name=='grud_common8':return ('input_linear','mask_linear','decay_hidden','classifier')
    if name in CONTROLS:return ('motion','antenna_encode','set_encode','blocks.0','blocks.1','classifier')
    from . import baseline_original as original,baseline_expanded as expanded
    return original.core_paths(name) if name in original.NAMES else expanded.core_paths(name)
