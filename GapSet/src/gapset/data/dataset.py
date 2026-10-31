"""One identical C1 observation-to-feature interface for every model."""

import json
from pathlib import Path

import numpy as np
import torch
from .contract import Observation, challenge
from torch.utils.data import Dataset

def encode(amplitude, valid, mean, std):
    # Hidden values cannot affect the public features or any fitted statistic.
    visible = np.where(valid, amplitude, 0).astype(np.float32)
    normalized = (np.log1p(visible) - mean) / std
    normalized[~valid] = 0
    steps, antennas, frequencies = visible.shape
    time = np.arange(steps, dtype=np.float32) / steps
    coordinate = np.arange(frequencies, dtype=np.float32) / max(1, frequencies - 1)
    image = np.concatenate(
        (
            normalized.transpose(1, 2, 0),
            valid.transpose(1, 2, 0).astype(np.float32),
            np.broadcast_to(time, (1, frequencies, steps)),
            np.broadcast_to(coordinate[None, :, None], (1, frequencies, steps)),
        ),
        axis=0,
    )
    if antennas != 3 or image.shape != (8, frequencies, steps) or not np.isfinite(image).all():
        raise ValueError("Invalid common feature contract")
    return image

class C1Dataset(Dataset):
    def __init__(self, cache, role, condition=None):
        self.cache = Path(cache)
        manifest = json.loads((self.cache / "manifest.json").read_text(encoding="utf-8"))
        if manifest["status"] != "ready":
            raise ValueError("Data gates not passed")
        self.rows = [row for row in manifest["records"] if row["role"] == role]
        self.manifest = {
            key: value for key, value in manifest.items() if key not in ("records", "rejected")
        }
        if not self.rows:
            raise ValueError("Empty split: " + role)
        self.condition = condition or {"name": "clean", "kind": "clean", "seed": 0}
        if role == "fit_clean" and self.condition["kind"] != "clean":
            raise ValueError("Clean training cannot use challenge masks")
        self.role = role
        self.amplitude = None
        self.valid = None

    def __getstate__(self):
        state = dict(self.__dict__)
        state["amplitude"] = state["valid"] = None
        return state

    def arrays(self):
        if self.amplitude is None:
            self.amplitude = np.memmap(
                self.cache / "amplitude.bin",
                mode="r",
                dtype=np.float32,
                shape=tuple(self.manifest["shape"]),
            )
            self.valid = np.memmap(
                self.cache / "valid.bin", mode="r", dtype=bool, shape=tuple(self.manifest["shape"])
            )

    def __len__(self):
        return len(self.rows)

    def observation(self, index):
        self.arrays()
        row = self.rows[index]
        x, valid = self.amplitude[row["row"]], self.valid[row["row"]]
        steps, antennas, frequencies = x.shape
        observation = Observation(
            x,
            valid,
            np.arange(steps, dtype=np.float64) * 0.01,
            np.arange(frequencies),
            np.arange(antennas),
            row["group_id"] + ":window" + str(row["offset_bins"]),
            "derived_grid",
        )
        c = self.condition
        if c["kind"] == "joint":
            for kind, amount, links in [
                ("rf_random", 0.3, ()),
                ("step_random", 0.3, ()),
                ("link_loss", 0.0, (0,)),
            ]:
                observation, _ = challenge(observation, kind, amount, seed=c["seed"], links=links)
        else:
            observation, _ = challenge(
                observation,
                c["kind"],
                c.get("amount", 0.0),
                seed=c["seed"],
                position=c.get("position", 0.5),
                links=tuple(c.get("links", [])),
            )
        return observation

    def __getitem__(self, index):
        observation = self.observation(index)
        normalizer = self.manifest["normalizer"]
        image = encode(
            observation.amplitude, observation.valid, normalizer["mean"], normalizer["std"]
        )
        return (
            torch.from_numpy(image),
            torch.from_numpy(observation.time_seconds.astype(np.float32)),
            self.rows[index]["label"],
            index,
        )

def conditions():
    panel = [{"name": "clean", "kind": "clean", "seed": 0}]
    for kind in ("rf_random", "step_random"):
        for amount in (0.1, 0.3, 0.5):
            for seed in (7, 19, 43):
                panel.append(
                    {
                        "name": f"{kind}_{int(amount * 100)}_s{seed}",
                        "kind": kind,
                        "amount": amount,
                        "seed": seed,
                    }
                )
    for retain in (0.75, 0.5, 0.25):
        for position in (0.0, 0.5, 1.0):
            panel.append(
                {
                    "name": f"frequency_retain{int(retain * 100)}_p{int(position * 100)}",
                    "kind": "frequency_span",
                    "amount": 1 - retain,
                    "position": position,
                    "seed": 0,
                }
            )
    for amount in (0.1, 0.3, 0.5):
        for position in (0.0, 0.5, 1.0):
            panel.append(
                {
                    "name": f"step_burst_{int(amount * 100)}_p{int(position * 100)}",
                    "kind": "step_burst",
                    "amount": amount,
                    "position": position,
                    "seed": 0,
                }
            )
    for links in ([0], [1], [2], [0, 1], [0, 2], [1, 2]):
        panel.append(
            {
                "name": "link_loss_" + "".join(map(str, links)),
                "kind": "link_loss",
                "links": links,
                "seed": 0,
            }
        )
    for seed in (7, 19, 43):
        panel.append({"name": f"joint_s{seed}", "kind": "joint", "seed": seed})
    return panel
