"""Analyze user-supplied grouped confusion counts, without model inference."""
import argparse
import json
from pathlib import Path
import numpy as np
from gapset.analysis.paired_groups import paired_intervals

def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    with np.load(a.input,allow_pickle=False) as z:
        result=paired_intervals(z['counts'],z['group_users'],z['models'].tolist(),z['conditions'].tolist())
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x',encoding='utf8') as f:json.dump(result,f,indent=2)

if __name__=='__main__':main()
