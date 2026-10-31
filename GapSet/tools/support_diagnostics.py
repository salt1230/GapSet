"""Optional fixed-seed42 diagnostic; does not run during package verification."""
import argparse
import os
from pathlib import Path
import time
from gapset.analysis.support_diagnostics import main as run
from gapset.cli import single_gpu_lock

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--max-hours',type=float,default=2);p.add_argument('--lock',type=Path,default=Path('outputs/.gpu.lock'))
    a=p.parse_args()
    if not 0<a.max_hours<=24:p.error('--max-hours must be in (0,24]')
    os.environ['GAPSET_RUN_DEADLINE']=str(time.time()+a.max_hours*3600)
    with single_gpu_lock(a.lock):run(a.plan)
