"""Full CPU input identity, disjointness and fit-only-statistic gate before CUDA."""

import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from .preprocess import digest, identity, save_json

def audit(cache, output, raw_root):
    cache, output = Path(cache), Path(output)
    manifest = json.loads((cache / "manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "ready" or manifest["target_test_enabled"] is not False:
        raise ValueError("Raw data completion/target scope failed")
    expected = manifest["files"]
    for name, sha in expected.items():
        if digest(cache / name) != sha:
            raise ValueError("Cache SHA mismatch: " + name)
    shape = tuple(manifest["shape"])
    if shape[1:] != (256, 3, 30):
        raise ValueError("Native shape/time contract changed")
    if (cache / "amplitude.bin").stat().st_size != np.prod(shape) * 4 or (
        cache / "valid.bin"
    ).stat().st_size != np.prod(shape):
        raise ValueError("Incomplete binary input")
    x = np.memmap(cache / "amplitude.bin", mode="r", dtype=np.float32, shape=shape)
    valid = np.memmap(cache / "valid.bin", mode="r", dtype=bool, shape=shape)
    rows = manifest["records"]
    if [r["row"] for r in rows] != list(range(shape[0])) or len({r["id"] for r in rows}) != len(
        rows
    ):
        raise ValueError("Row identity not unique/complete")
    group_roles, raw_roles, canonical_roles = {}, {}, {}
    fingerprint = hashlib.sha256()
    totals = [0.0, 0.0, 0]
    for row in rows:
        expected_identity = identity(Path(row["raw_path"]), Path(raw_root).resolve())
        if expected_identity is None or any(
            row[key] != expected_identity[key] for key in expected_identity
        ):
            raise ValueError("Raw metadata split/label changed")
        for mapping, key in ((group_roles, row["group_id"]), (raw_roles, row["raw_sha256"])):
            if mapping.setdefault(key, row["role"]) != row["role"]:
                raise ValueError("Group/raw content overlaps internal or dev sets")
        amplitude, mask = x[row["row"]], valid[row["row"]]
        if (
            not mask.any()
            or not np.isfinite(amplitude[mask]).all()
            or np.any(amplitude[mask] < 0)
            or not np.any(amplitude[mask] > 0)
        ):
            raise ValueError("Invalid visible amplitude")
        canonical = hashlib.sha256(amplitude.tobytes() + mask.tobytes()).hexdigest()
        if canonical_roles.setdefault(canonical, row["role"]) != row["role"]:
            raise ValueError("Identical canonical observations cross partitions")
        fingerprint.update(row["id"].encode())
        fingerprint.update(canonical.encode())
        if row["role"] == "fit_clean":
            values = np.log1p(amplitude[mask].astype(np.float64))
            totals[0] += values.sum()
            totals[1] += np.dot(values, values)
            totals[2] += len(values)
    mean = totals[0] / totals[2]
    std = max(np.sqrt(max(0.0, totals[1] / totals[2] - mean**2)), 1e-6)
    if not np.allclose(
        [mean, std],
        [manifest["normalizer"]["mean"], manifest["normalizer"]["std"]],
        atol=1e-9,
        rtol=1e-7,
    ):
        raise ValueError("Source-only normalization independently disagrees")
    counts = {
        role: dict(Counter(r["label"] for r in rows if r["role"] == role))
        for role in ("fit_clean", "source_val_clean", "cross_domain_dev")
    }
    if any(set(count) != set(range(6)) for count in counts.values()):
        raise ValueError("Six-class coverage gate failed")
    record = {
        "status": "passed",
        "manifest_sha256": digest(cache / "manifest.json"),
        "files": expected,
        "shape": list(shape),
        "role_counts": manifest["role_counts"],
        "class_counts": counts,
        "rejected_raw_files": manifest["rejected"],
        "canonical_identity_sha256": fingerprint.hexdigest(),
        "source_normalizer_independently_recomputed": {"mean": mean, "std": std},
        "raw_content_and_canonical_content_disjoint": True,
        "whole_recording_group_disjoint": True,
        "raw_users_not_disjoint_between_fit_and_source_val": True,
        "final_test_accessed": False,
        "cross_domain_is_development_previously_accessed": True,
        "cuda_used": False,
    }
    save_json(output, record)
    print(json.dumps({k: record[k] for k in ("status", "shape", "role_counts")}), flush=True)
    return record
