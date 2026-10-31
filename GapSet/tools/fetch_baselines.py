"""Fetch explicit pinned author files without executing or installing them."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT=Path(__file__).resolve().parents[1]

def fetch(record,output):
    target=output/record['target']
    if target.exists():
        if hashlib.sha256(target.read_bytes()).hexdigest()!=record['sha256']:raise ValueError('Existing file differs: '+str(target))
        return
    with urllib.request.urlopen(record['url'],timeout=30) as response:data=response.read(2*1024**2)
    if hashlib.sha256(data).hexdigest()!=record['sha256']:raise ValueError('Pinned source checksum mismatch')
    target.parent.mkdir(parents=True,exist_ok=True)
    with target.open('xb') as f:f.write(data)

def main():
    p=argparse.ArgumentParser(description='Optional third-party sources retain their upstream terms; see THIRD_PARTY.md')
    p.add_argument('--component',choices=['wiflexformer','csibert2','amcs','hms','all'],required=True)
    p.add_argument('--output',type=Path,default=ROOT/'third_party')
    a=p.parse_args();rows=json.loads((ROOT/'third_party/sources.lock.json').read_text())
    for r in rows:
        if a.component in ('all',r['component']):fetch(r,a.output);print('verified',r['target'])

if __name__=='__main__':main()
