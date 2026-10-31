"""Real amplitude adapters. Raw preparation uses a separate csiread overlay."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import time

import numpy as np

DATES = {'20181204': {1}, '20181209': {2, 6}, '20181211': {3, 7, 8, 9},
         '20181130': {5, 10, 11, 12, 13, 14, 15, 16, 17}}
NAME = re.compile(r'^user(\d+)-(\d+)-(\d+)-(\d+)-(\d+)-r(\d+)\.dat$')
T = 256
BIN_US = 10000

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()

def save_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.pending')
    temp.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temp.replace(path)

def identity(path, raw):
    found = NAME.match(path.name)
    if not found:
        return None
    date = next((p for p in path.relative_to(raw).parts if p in DATES), None)
    user, label, location, orientation, repeat, rx = map(int, found.groups())
    if date is None or user not in DATES[date] or not 1 <= label <= 6 or not 1 <= rx <= 6:
        return None
    group = f'{date}:user{user}-{label}-{location}-{orientation}-{repeat}'
    uniform = int.from_bytes(hashlib.sha256(('42:' + group).encode()).digest()[:8], 'big') / 2**64
    role = 'cross_domain_dev' if date == '20181130' else ('source_val_clean' if uniform < .2 else 'fit_clean')
    return {'raw_path': str(path), 'group_id': group, 'date': date, 'user': user,
            'label': label - 1, 'location': location, 'orientation': orientation,
            'repeat': repeat, 'receiver': rx, 'role': role}

def aggregate_packets(amplitude, ticks):
    """Mean linear amplitude in elapsed 10ms bins, without stretching time."""
    ticks = np.asarray(ticks, dtype=np.int64)
    unwrapped = ticks + np.r_[0, np.cumsum(np.diff(ticks) < -(2**31))] * 2**32
    if len(ticks) < 2 or np.any(np.diff(unwrapped) <= 0):
        raise ValueError('non_increasing_packet_counter')
    bins = ((unwrapped - unwrapped[0]) // BIN_US).astype(np.int64)
    if bins[-1] > 6000:
        raise ValueError('record_exceeds_prespecified_60seconds')
    begins = np.r_[0, np.flatnonzero(np.diff(bins)) + 1]
    counts = np.diff(np.r_[begins, len(bins)])
    values = np.add.reduceat(amplitude.astype(np.float64), begins, axis=0) / counts[:, None, None]
    result = np.zeros((int(bins[-1]) + 1, 3, 30), dtype=np.float32)
    valid = np.zeros(result.shape, dtype=bool)
    result[bins[begins]] = values.astype(np.float32)
    valid[bins[begins]] = True
    return result, valid, (unwrapped[-1] - unwrapped[0]) / 1e6

def window_array(amplitude, valid):
    """Nonoverlapping windows; all windows stay tied to the original recording."""
    result = []
    for offset in range(0, len(amplitude), T):
        stop = min(offset + T, len(amplitude))
        if not valid[offset:stop].any():
            continue
        x = np.zeros((T, *amplitude.shape[1:]), dtype=np.float32)
        m = np.zeros(x.shape, dtype=bool)
        x[:stop-offset] = amplitude[offset:stop]
        m[:stop-offset] = valid[offset:stop]
        result.append((offset, x, m))
    return result

def parse_raw(row):
    import csiread

    row = dict(row)
    path = Path(row['raw_path'])
    row['raw_sha256'] = digest(path)
    try:
        reader = csiread.Intel(str(path), nrxnum=3, ntxnum=1, pl_size=10, if_report=False)
        reader.read()
        if np.any(reader.rate & 0x800):
            raise ValueError('non20MHz_rate_flag')
        if np.any(reader.Ntx != 1) or np.any(reader.Nrx != 3):
            raise ValueError('not_three_receive_one_transmit')
        csi = reader.get_scaled_csi()
        if csi.ndim != 4 or csi.shape[1:] != (30, 3, 1) or not np.isfinite(csi).all():
            raise ValueError('invalid_scaled_csi')
        amplitude = np.abs(csi[..., 0]).transpose(0, 2, 1)
        x, m, duration = aggregate_packets(amplitude, reader.timestamp_low)
        row.update(raw_packets=len(csi), duration_seconds=duration,
                   missing_natural_bins=int((~m.any(axis=(1, 2))).sum()))
        return {'record': row, 'windows': window_array(x, m)}
    except Exception as exc:
        return {'record': row, 'rejected': str(exc)}

def prepare(raw, output, workers=2):
    raw, output = Path(raw).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for date in DATES:
        if not (raw/date).is_dir():
            raise FileNotFoundError(date)
        for path in sorted((raw/date).rglob('*.dat')):
            row = identity(path, raw)
            if row:
                rows.append(row)
    if not rows:
        raise ValueError('Empty raw inventory')
    save_json(output/'inventory.json', rows)
    accepted, rejected = [], []
    content_roles, group_roles = {}, {}
    start = time.monotonic()
    sums = [0., 0., 0]
    with (output/'amplitude.bin').open('xb') as xs, (output/'valid.bin').open('xb') as ms:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for index, result in enumerate(pool.map(parse_raw, rows, chunksize=8)):
                row = result['record']
                if 'rejected' in result:
                    rejected.append({**row, 'reason': result['rejected']})
                else:
                    digest_role = content_roles.setdefault(row['raw_sha256'], row['role'])
                    group_role = group_roles.setdefault(row['group_id'], row['role'])
                    if digest_role != row['role'] or group_role != row['role']:
                        raise ValueError('Content/recording group crosses partitions')
                    for offset, x, m in result['windows']:
                        record = {**row, 'row': len(accepted), 'offset_bins': offset,
                                  'id': f"{row['group_id']}:rx{row['receiver']}:w{offset}"}
                        xs.write(x.tobytes()); ms.write(m.tobytes())
                        accepted.append(record)
                        if row['role'] == 'fit_clean':
                            visible = np.log1p(x[m].astype(np.float64))
                            sums[0] += float(visible.sum())
                            sums[1] += float(np.dot(visible, visible))
                            sums[2] += int(len(visible))
                if (index+1) % 250 == 0:
                    save_json(output/'progress.json', {'phase': 'raw_parse', 'parsed': index+1,
                              'total': len(rows), 'windows': len(accepted), 'rejected': len(rejected),
                              'elapsed_seconds': time.monotonic()-start})
                    print(json.dumps({'parsed': index+1, 'total': len(rows), 'windows': len(accepted)}), flush=True)
    role_counts = Counter(r['role'] for r in accepted)
    class_counts = {role: dict(Counter(r['label'] for r in accepted if r['role'] == role)) for role in role_counts}
    for role in ('fit_clean', 'source_val_clean', 'cross_domain_dev'):
        if set(class_counts.get(role, {})) != set(range(6)):
            raise ValueError('Missing class coverage in '+role)
    rejection_fraction = len(rejected)/len(rows)
    mean = sums[0]/sums[2]
    std = max((sums[1]/sums[2]-mean**2)**.5, 1e-6)
    status = 'ready' if rejection_fraction <= .01 else 'rejection_gate_failed'
    manifest = {'status': status, 'scope': 'six_class_raw_amplitude_development',
                'records': accepted, 'rejected': rejected, 'raw_files': len(rows),
                'rejection_fraction': rejection_fraction, 'role_counts': dict(role_counts),
                'class_counts': class_counts, 'shape': [len(accepted), T, 3, 30],
                'bin_us': BIN_US, 'window_steps': T, 'frequency_coordinates': 'reported_Intel_group_ordinal',
                'time_semantics': 'grid_derived_from_measured_microsecond_counter',
                'normalizer': {'fit_role': 'fit_clean', 'transform': 'log1p', 'mean': mean, 'std': std},
                'history': '20181130 has prior development test access; NOT untouched final test',
                'target_test_enabled': False, 'room_mapping_claimed': False,
                'files': {name: digest(output/name) for name in ('amplitude.bin', 'valid.bin', 'inventory.json')},
                'elapsed_seconds': time.monotonic()-start}
    save_json(output/'manifest.json', manifest)
    save_json(output/'completion.json', {k: manifest[k] for k in ('status', 'role_counts', 'class_counts', 'elapsed_seconds')})
    print(json.dumps(manifest['role_counts']), flush=True)
    if status != 'ready':
        raise ValueError('More than1% raw input exclusions; inspect evidence before training')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--workers', type=int, default=2, choices=[1, 2])
    args = parser.parse_args()
    prepare(args.raw, args.output, args.workers)
