"""C1 native-frequency amplitude contract; no interpolation or bandwidth synthesis.

This is an input/mask primitive, not an approved full preprocessing pipeline.
All perturbations operate on published observation steps; packet challenges need
a verified raw-packet adapter before aggregation.
"""

import hashlib
import json
from dataclasses import dataclass, replace

import numpy as np

@dataclass(frozen=True)
class Observation:
    amplitude: np.ndarray  # T,A,F
    valid: np.ndarray
    time_seconds: np.ndarray
    frequency_id: np.ndarray
    link_id: np.ndarray
    group_id: str
    time_semantics: str

    def validate(self):
        x, m = self.amplitude, self.valid
        if x.ndim != 3 or m.shape != x.shape or m.dtype != np.bool_:
            raise ValueError("Expected amplitude[T,A,F] and bool valid of identical shape")
        t, a, f = x.shape
        if a != 3 or f not in (30, 114) or t < 2:
            raise ValueError("C1 draft expects 3 antenna pairs and native 30/114 RF positions")
        if self.time_seconds.shape != (t,) or not np.isfinite(self.time_seconds).all():
            raise ValueError("Invalid relative timestamps")
        if np.any(np.diff(self.time_seconds) <= 0):
            raise ValueError("Times must increase; missing observations must not compress time")
        if self.frequency_id.shape != (f,) or len(np.unique(self.frequency_id)) != f:
            raise ValueError("RF IDs must be unique and retain their original identity")
        if self.link_id.shape != (a,) or len(np.unique(self.link_id)) != a:
            raise ValueError("Link IDs must be unique")
        if not np.isfinite(x[m]).all() or np.any(x[m] < 0):
            raise ValueError("Visible amplitude must be finite and nonnegative")
        if self.time_semantics not in ("measured", "nominal_published", "derived_grid") or not self.group_id:
            raise ValueError("Missing measurement provenance")
        return self

    def visible_values(self):
        self.validate()
        return np.where(self.valid, self.amplitude, 0).astype(np.float32)

def challenge(observation, kind, amount=0.0, *, seed=7, position=0.5, links=()):
    """Deterministic label-blind masks, applied to actual available support.

    Ratio denominator excludes existing invalid cells. RF masks use positions
    with at least one visible value; step masks use actually observed steps.
    Contiguous masks retain their original locations; they never close gaps.
    """
    observation.validate()
    if not 0 <= amount < 1 or not 0 <= position <= 1:
        raise ValueError("Invalid challenge strength/position")
    identity = json.dumps(
        [observation.group_id, kind, amount, seed, position, list(links)], separators=(",", ":")
    )
    digest = hashlib.sha256(identity.encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
    m = observation.valid.copy()
    if kind == "clean":
        if amount or links:
            raise ValueError("Clean cannot include a challenge")
    elif kind in ("rf_random", "frequency_span"):
        available = np.flatnonzero(m.any(axis=(0, 1)))
        if not len(available):
            raise ValueError("No RF observations")
        if kind == "rf_random":
            count = int(np.floor(len(available) * amount))
            m[:, :, rng.choice(available, count, replace=False)] = False
        else:
            # amount is the fraction discarded; no assertion about RF bandwidth.
            width = max(1, int(np.ceil(len(available) * (1 - amount))))
            start = round((len(available) - width) * position)
            keep = available[start : start + width]
            lost = np.setdiff1d(available, keep)
            m[:, :, lost] = False
    elif kind in ("step_random", "step_burst"):
        available = np.flatnonzero(m.any(axis=(1, 2)))
        if not len(available):
            raise ValueError("No observed steps")
        count = int(np.floor(len(available) * amount))
        if kind == "step_random":
            lost = rng.choice(available, count, replace=False)
        else:
            # One interval on the original time-index axis; no compressed sequence.
            width = int(np.floor((available[-1] - available[0] + 1) * amount))
            start = round(available[0] + (available[-1] - available[0] + 1 - width) * position)
            lost = np.arange(start, start + width)
        m[lost, :, :] = False
    elif kind == "link_loss":
        if not links or len(set(links)) != len(links) or len(links) >= m.shape[1]:
            raise ValueError("Drop one/two distinct antenna pairs, never all pairs")
        if any(not 0 <= link < m.shape[1] for link in links):
            raise ValueError("Unknown link")
        m[:, list(links), :] = False
    else:
        raise ValueError("Unknown challenge: " + kind)
    result = replace(observation, valid=m).validate()
    audit = {
        "kind": kind,
        "requested_amount": amount,
        "seed": seed,
        "position": position,
        "links": list(links),
        "group_id": observation.group_id,
        "baseline_visible_cells": int(observation.valid.sum()),
        "remaining_visible_cells": int(m.sum()),
        "additional_missing_cells": int((observation.valid & ~m).sum()),
        "mask_sha256": hashlib.sha256(np.ascontiguousarray(m).tobytes()).hexdigest(),
        "semantic": "published_observation_step_or_RF_support_not_hardware_bandwidth",
        "time_unchanged": True,
        "coordinate_unchanged": True,
        "labels_accessed": False,
    }
    return result, audit

@dataclass(frozen=True)
class SourceNormalizer:
    mean: float
    std: float

    @classmethod
    def fit(cls, observations):
        # Source-fit caller is responsible for the disjoint training manifest.
        total = 0
        sums = 0.0
        squares = 0.0
        for obs in observations:
            obs.validate()
            value = np.log1p(obs.amplitude[obs.valid].astype(np.float64))
            total += value.size
            sums += value.sum()
            squares += np.square(value).sum()
        if not total:
            raise ValueError("No source-fit measurements")
        mean = float(sums / total)
        return cls(mean, float(max(squares / total - mean * mean, 1e-12) ** 0.5))

    def transform(self, observation):
        visible = observation.visible_values()
        normalized = (np.log1p(visible) - self.mean) / self.std
        return np.where(observation.valid, normalized, 0).astype(np.float32)
