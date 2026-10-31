"""Fresh clean training, source validation selection, then fixed challenge eval."""

import argparse

import hashlib

import json

import random

import sys

import time

from pathlib import Path

import numpy as np

import torch

from gapset.data.dataset import C1Dataset, conditions

from .guard import HEADROOM, Guard, ResourcePause

from gapset.metrics import metrics

from gapset.models.registry import NAMES, YEARS, core_paths, make_model

from torch import nn

from torch.utils.data import DataLoader, RandomSampler

from gapset.data.preprocess import digest, save_json

SEED = 42

MICRO = 128

EFFECTIVE = 512

ACCUM = EFFECTIVE // MICRO

EPOCHS = 200

CLASSES = 6

def cpu_state(value):

    if isinstance(value, torch.Tensor):

        return value.detach().cpu().clone()

    if isinstance(value, dict):

        return {key: cpu_state(v) for key, v in value.items()}

    if isinstance(value, list):

        return [cpu_state(v) for v in value]

    if isinstance(value, tuple):

        return tuple(cpu_state(v) for v in value)

    return value

def checkpoint(path, state):

    pending = path.with_suffix(path.suffix + ".pending")

    torch.save(cpu_state(state), pending)

    pending.replace(path)

def model_fingerprint(model):

    value = hashlib.sha256()

    for name, tensor in model.state_dict().items():

        value.update(name.encode())

        value.update(str(tensor.dtype).encode())

        value.update(tensor.detach().cpu().contiguous().numpy().tobytes())

    return value.hexdigest()

def seed():

    random.seed(SEED)

    np.random.seed(SEED)

    torch.manual_seed(SEED)

    torch.cuda.manual_seed_all(SEED)

    torch.set_num_threads(2)

    torch.backends.cuda.matmul.allow_tf32 = False

    torch.backends.cudnn.allow_tf32 = False

    torch.backends.cudnn.benchmark = False

def loader(dataset, *, shuffle=False, generator=None):

    sampler = RandomSampler(dataset, generator=generator) if shuffle else None

    return DataLoader(

        dataset,

        batch_size=MICRO,

        sampler=sampler,

        shuffle=False,

        drop_last=False,

        num_workers=2,

        pin_memory=True,

        persistent_workers=True,

        multiprocessing_context="spawn",

        generator=torch.Generator().manual_seed(SEED+4200),

    )

def preimport(name):

    if name.startswith('ours_'):

        from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

        assert callable(selective_scan_fn)

def initialize(args):

    args.output.mkdir(parents=True, exist_ok=True)

    identity = json.loads(args.input_identity.read_text(encoding="utf-8"))

    if identity["status"] != "passed" or identity["manifest_sha256"] != digest(

        args.data / "manifest.json"

    ):

        raise ValueError("Frozen input identity changed or not passed")

    preimport(args.model)

    seed()

    guard = Guard(args.output)

    guard.admit()

    model = make_model(args.model, args.vendor).cuda()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=0.0)

    return model, optimizer, scheduler, guard

def run_inference(model, data_loader, guard):

    model.eval()

    all_logits, all_labels, ids = [], [], []

    losses = 0.0

    with torch.no_grad():

        for image, timestamp, label, index in data_loader:

            started = time.monotonic()

            guard.check()

            output = model(image.cuda(non_blocking=True), timestamp.cuda(non_blocking=True))

            if not torch.isfinite(output).all():

                raise ValueError("Nonfinite logits")

            losses += nn.functional.cross_entropy(output, label.cuda(), reduction="sum").item()

            all_logits.append(output.cpu().numpy())

            all_labels.append(label.numpy())

            ids.append(index.numpy())

            guard.throttle(started)

    logits, labels, indices = (

        np.concatenate(all_logits),

        np.concatenate(all_labels),

        np.concatenate(ids),

    )

    if not np.array_equal(indices, np.arange(len(labels))):

        raise ValueError("Prediction order does not match held-out identities")

    result = metrics(labels, logits.argmax(1), CLASSES)

    matrix = np.bincount(labels * CLASSES + logits.argmax(1), minlength=CLASSES**2).reshape(

        CLASSES, CLASSES

    )

    result.update(

        loss=losses / len(labels),

        confusion=matrix.tolist(),

        class_recall=np.divide(

            np.diag(matrix), matrix.sum(1), out=np.zeros(CLASSES), where=matrix.sum(1) > 0

        ).tolist(),

    )

    return logits, labels, result

def gate(args):

    if (args.output / "gate.json").exists():

        raise ValueError("A passed gate must not be overwritten")

    model, optimizer, _, guard = initialize(args)

    initial_fingerprint = model_fingerprint(model)

    if initial_fingerprint != json.loads(args.cpu_gate.read_text())["initial_model_sha256"]:

        raise ValueError("Actual CPU/CUDA initial-state pin mismatch")

    from .numerical_gate import check

    check(args, model, guard)

    fit = C1Dataset(args.data, "fit_clean")

    generator = torch.Generator().manual_seed(SEED)

    batch_loader = loader(fit, shuffle=True, generator=generator)

    modules = dict(model.named_modules())

    calls = {name: 0 for name in core_paths(args.model)}

    handles = [

        modules[name].register_forward_hook(

            lambda module, inputs, output, name=name: calls.__setitem__(name, calls[name] + 1)

        )

        for name in calls

    ]

    optimizer.zero_grad(set_to_none=True)

    model.train()

    begin = time.monotonic()

    examples = []

    for step, (image, timestamp, labels, _) in enumerate(batch_loader):

        if step == ACCUM:

            begin = time.monotonic()

        started = time.monotonic()

        image, timestamp, labels = image.cuda(), timestamp.cuda(), labels.cuda()

        logits = model(image, timestamp)

        loss = nn.functional.cross_entropy(logits, labels) * len(labels) / EFFECTIVE

        loss.backward()

        if not torch.isfinite(logits).all() or not torch.isfinite(loss):

            raise ValueError("Nonfinite actual-input gate")

        if step >= ACCUM:

            examples.append((image, timestamp))

        guard.throttle(started)

        if step == ACCUM - 1:

            optimizer.zero_grad(set_to_none=True)

        if step == 2 * ACCUM - 1:

            break

    wall_seconds = time.monotonic() - begin

    gradients = {

        name: parameter.grad

        for name, parameter in model.named_parameters()

        if parameter.grad is not None

    }

    if not gradients or not all(torch.isfinite(value).all() for value in gradients.values()):

        raise ValueError("Nonfinite/missing backward")

    for name, count in calls.items():

        if count == 0:

            raise ValueError("Core component never executed: " + name)

        module_parameters = list(modules[name].parameters())

        if module_parameters and not any(

            p.grad is not None and p.grad.abs().max().item() > 0 for p in module_parameters

        ):

            raise ValueError("Core component receives no learning signal: " + name)

    full_params = sum(p.numel() for p in model.parameters())

    active_params = sum(p.numel() for p in model.parameters() if p.grad is not None)

    before = {

        name: value.detach().cpu().clone()

        for name, value in model.named_parameters()

        if value.grad is not None

    }

    nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    optimizer.step()

    if not all(torch.isfinite(value).all() for value in model.parameters()):

        raise ValueError("Nonfinite isolated AdamW step")

    changed = any(

        not torch.equal(before[name], value.detach().cpu())

        for name, value in model.named_parameters()

        if name in before

    )

    if not changed:

        raise ValueError("Isolated AdamW step changed no active parameters")

    model.eval()

    image, timestamp = examples[0]

    with torch.no_grad():

        full = model(image, timestamp)

        split = torch.cat(

            [model(image[i : i + 32], timestamp[i : i + 32]) for i in range(0, len(image), 32)]

        )

    parity = (full - split).abs().max().item()

    if parity > 2e-4:

        raise ValueError("Eval microbatch parity failed: " + str(parity))

    checkpoint(

        args.output / "isolated_checkpoint.pth",

        {"model": model.state_dict(), "optimizer": optimizer.state_dict()},

    )

    restored_state = torch.load(

        args.output / "isolated_checkpoint.pth", map_location="cpu", weights_only=False

    )

    # Discard the gate model before reload: no duplicate large MM-Fi MLP CUDA states.

    for handle in handles:

        handle.remove()

    del model, optimizer, modules, gradients, module_parameters, logits, loss

    torch.cuda.empty_cache()

    restored = make_model(args.model, args.vendor).cuda().eval()

    restored.load_state_dict(restored_state["model"], strict=True)

    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-4, weight_decay=1e-3)

    restored_optimizer.load_state_dict(restored_state["optimizer"])

    with torch.no_grad():

        reload_difference = (full - restored(image, timestamp)).abs().max().item()

    if reload_difference > 1e-6:

        raise ValueError("Isolated checkpoint strict reload failed")

    model = restored

    del restored_optimizer, restored_state

    latencies = []

    with torch.no_grad():

        for iteration in range(15):

            started = time.monotonic()

            begin_event, end_event = (

                torch.cuda.Event(enable_timing=True),

                torch.cuda.Event(enable_timing=True),

            )

            begin_event.record()

            model(image[:1], timestamp[:1])

            end_event.record()

            end_event.synchronize()

            if iteration >= 5:

                latencies.append(begin_event.elapsed_time(end_event))

            guard.throttle(started)

    guard.check(True)

    record = {

        "status": "passed",

        "actual_card_total_bytes": guard.snapshot()["total_bytes"],

        "seed": SEED,

        "model": args.model,

        "train_microbatch": MICRO,

        "initial_model_sha256": initial_fingerprint,

        "strict_checkpoint_reload_max_logit_difference": reload_difference,

        "effective_batch": EFFECTIVE,

        "steps_per_update": ACCUM,

        "eval_microbatch": MICRO,

        "actual_input_shape": list(image.shape),

        "logical_batch_wall_seconds": wall_seconds,

        "isolated_step_discarded": True,

        "core_forward_calls": calls,

        "params_full_registered": full_params,

        "params_receiving_gradients": active_params,

        "eval_128_vs32_max_logit_difference": parity,

        "batch1_cuda_median_ms": float(np.median(latencies)),

        "batch1_cuda_p90_ms": float(np.quantile(latencies, 0.9)),

        "flops_status": "not_claimed; full operator accounting pending, no partial count labelled whole FLOPs",

        "shared_memory_budget_including_headroom_bytes": max(

            torch.cuda.max_memory_reserved(), guard.snapshot()["used_bytes"]

        )

        + HEADROOM,

        "estimated_train_epoch_seconds": wall_seconds * np.ceil(len(fit) / EFFECTIVE),

        "input_manifest_sha256": digest(args.data / "manifest.json"),

    }

    save_json(args.output / "gate.json", record)

    print(json.dumps(record), flush=True)

def train(args):

    if (args.output / "completion.json").exists():

        raise ValueError("Completed case must not retrain")

    model, optimizer, scheduler, guard = initialize(args)

    gate_record = json.loads((args.output.parent / "gate/gate.json").read_text())

    if model_fingerprint(model) != gate_record["initial_model_sha256"]:

        raise ValueError(

            "Fresh initialization differs from passed gate; no smoke/checkpoint migration"

        )

    fit, val = C1Dataset(args.data, "fit_clean"), C1Dataset(args.data, "source_val_clean")

    generator = torch.Generator().manual_seed(SEED)

    training, validation = loader(fit, shuffle=True, generator=generator), loader(val)

    recipe = {

        "model": args.model,

        "seed": SEED,

        "optimizer": "AdamW",

        "lr": 1e-4,

        "weight_decay": 1e-3,

        "train_effective_batch": EFFECTIVE,

        "physical_microbatch": MICRO,

        "eval_microbatch": MICRO,

        "gradient_accumulation": ACCUM,

        "epochs": EPOCHS,

        "early_stop": False,

        "augmentation": False,

        "selection_split": "source_val_clean",

        "selection_metric": "macro_f1_percent",

        "input_manifest_sha256": digest(args.data / "manifest.json"),

        "dropout_note": "Model constructor retained; HMS0.15, GRU-D0, GapSet0.1; see model source",

        "precision": "float32_tf32_disabled",

        "core_architecture_source_fixed": True,

        "batch_norm_note": "Constructor-specific BN uses physical128; effective512 accumulation does not change BN statistics",

        "shuffle_rule": "independent RandomSampler declared_seed+epoch; worker RNG is separate",

        "external_pretraining": False,

        "method_identity": args.model,

    }

    if (args.output / "recipe.json").exists():

        if json.loads((args.output / "recipe.json").read_text()) != recipe:

            raise ValueError("Resume recipe changed")

    else:

        save_json(args.output / "recipe.json", recipe)

    start_epoch, best, history = 1, -1.0, []

    if (args.output / "last.pth").exists():

        state = torch.load(args.output / "last.pth", map_location="cpu", weights_only=False)

        if state["recipe"] != recipe:

            raise ValueError("Checkpoint recipe mismatch")

        model.load_state_dict(state["model"], strict=True)

        optimizer.load_state_dict(state["optimizer"])

        scheduler.load_state_dict(state["scheduler"])

        generator.set_state(state["loader_rng"])

        torch.set_rng_state(state["torch_rng"])

        torch.cuda.set_rng_state_all(state["cuda_rng"])

        np.random.set_state(state["numpy_rng"])

        random.setstate(state["python_rng"])

        start_epoch, best, history = state["epoch"] + 1, state["best_f1"], state["history"]

    for epoch in range(start_epoch, EPOCHS + 1):

        epoch_begin = time.monotonic()

        generator.manual_seed(SEED + epoch)

        model.train()

        optimizer.zero_grad(set_to_none=True)

        train_loss, train_correct, processed = 0.0, 0, 0

        lr = optimizer.param_groups[0]["lr"]

        for step, (image, timestamp, labels, _) in enumerate(training):

            started = time.monotonic()

            guard.check()

            image, timestamp, labels = (

                image.cuda(non_blocking=True),

                timestamp.cuda(non_blocking=True),

                labels.cuda(),

            )

            output = model(image, timestamp)

            raw_loss = nn.functional.cross_entropy(output, labels)

            # Final smaller logical batch is normalized to its true example count.

            group_start = (step // ACCUM) * EFFECTIVE

            group_size = min(EFFECTIVE, len(fit) - group_start)

            loss = raw_loss * len(labels) / group_size

            if not torch.isfinite(loss):

                raise ValueError("Nonfinite training loss")

            loss.backward()

            if step % ACCUM == ACCUM - 1 or step == len(training) - 1:

                norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                if not torch.isfinite(norm):

                    raise ValueError("Nonfinite training gradient")

                optimizer.step()

                optimizer.zero_grad(set_to_none=True)

            train_loss += raw_loss.item() * len(labels)

            train_correct += int((output.argmax(1) == labels).sum().item())

            processed += len(labels)

            guard.throttle(started)

            if step % 20 == 0:

                save_json(

                    args.output / "progress.json",

                    {

                        "phase": "train",

                        "epoch": epoch,

                        "step": step + 1,

                        "steps": len(training),

                        "last_complete_epoch": epoch - 1,

                        "time_unix": time.time(),

                    },

                )

        logits, labels, result = run_inference(model, validation, guard)

        if processed != len(fit):

            raise ValueError("Training sample count changed")

        result.update(

            epoch=epoch,

            train_loss=train_loss / processed,

            train_accuracy_percent=train_correct / processed * 100,

            lr=lr,

            epoch_seconds=time.monotonic() - epoch_begin,

        )

        chosen = result["macro_f1_percent"] > best

        if chosen:

            best = result["macro_f1_percent"]

        history.append(result)

        scheduler.step()

        state = {

            "epoch": epoch,

            "model": model.state_dict(),

            "optimizer": optimizer.state_dict(),

            "scheduler": scheduler.state_dict(),

            "loader_rng": generator.get_state(),

            "torch_rng": torch.get_rng_state(),

            "cuda_rng": torch.cuda.get_rng_state_all(),

            "numpy_rng": np.random.get_state(),

            "python_rng": random.getstate(),

            "best_f1": best,

            "history": history,

            "recipe": recipe,

        }

        np.savez_compressed(

            args.output / f"epoch_{epoch:03d}_source_val.npz",

            labels=labels,

            logits=logits,

            row_indices=np.array([r["row"] for r in val.rows], dtype=np.int64),

        )

        save_json(args.output / f"epoch_{epoch:03d}.json", result)

        if chosen:

            checkpoint(args.output / "best.pth", state)

        checkpoint(args.output / "last.pth", state)

        save_json(args.output / "curves.json", history)

        save_json(

            args.output / "progress.json",

            {

                "phase": "epoch_committed",

                "last_complete_epoch": epoch,

                "best_source_f1_percent": best,

                "time_unix": time.time(),

            },

        )

        print(json.dumps(result), flush=True)

    save_json(

        args.output / "completion.json",

        {

            "status": "complete",

            "epochs": EPOCHS,

            "selection_split": "source_val_clean",

            "best_macro_f1_percent": best,

            "target_test_accessed": False,

            "completed_at_unix": time.time(),

        },

    )

def evaluate(args):

    model, _, _, guard = initialize(args)

    state = torch.load(args.training / "best.pth", map_location="cpu", weights_only=False)

    if state["recipe"]["selection_split"] != "source_val_clean":

        raise ValueError("Mixed checkpoint selection")

    model.load_state_dict(state["model"], strict=True)

    records = []

    for role in ("source_val_clean", "cross_domain_dev"):

        for c in conditions():

            folder = args.output / role / c["name"]

            done = folder / "metrics.json"

            if done.exists():

                record = json.loads(done.read_text())

                for key in ("label_file", "prediction_file", "logit_file"):

                    if digest(record[key]) != record[key + "_sha256"]:

                        raise ValueError("Existing evaluation artifact changed")

                records.append(record)

                continue

            folder.mkdir(parents=True, exist_ok=True)

            dataset = C1Dataset(args.data, role, c)

            logits, labels, result = run_inference(model, loader(dataset), guard)

            # One deterministic mask audit over the entire evaluated sample list.

            mask_digest = hashlib.sha256()

            visible, original = 0, 0

            dataset.arrays()

            for index, row in enumerate(dataset.rows):

                observation = dataset.observation(index)

                mask_digest.update(row["id"].encode())

                mask_digest.update(observation.valid.tobytes())

                visible += int(observation.valid.sum())

                original += int(dataset.valid[row["row"]].sum())

            np.save(folder / "labels.npy", labels)

            np.save(folder / "predictions.npy", logits.argmax(1))

            np.save(folder / "logits.npy", logits)

            # Record-level aggregation is supplementary; all correlated RX remain grouped.

            scores = np.exp(logits - logits.max(1, keepdims=True))

            scores /= scores.sum(1, keepdims=True)

            frozen_masks = json.loads(args.expected_masks.read_text())[

                "tests"

            ]

            frozen_mask = next(

                r for r in frozen_masks if r["split"] == role and r["condition"] == c["name"]

            )

            if mask_digest.hexdigest() != frozen_mask["mask_sha256"]:

                raise ValueError("Mask differs from pre-score frozen Widar challenge")

            groups = {}

            for row, score in zip(dataset.rows, scores, strict=True):

                key = row["group_id"] + ":rx" + str(row["receiver"])

                groups.setdefault(key, {"label": row["label"], "scores": []})["scores"].append(

                    score

                )

            record_metrics = metrics(

                np.array([v["label"] for v in groups.values()]),

                np.array([np.stack(v["scores"]).mean(0).argmax() for v in groups.values()]),

                CLASSES,

            )

            result.update(

                model=args.model,

                year=YEARS[args.model],

                split=role,

                condition=c["name"],

                challenge=c,

                training_seed=SEED,

                challenge_seed=c["seed"],

                selection_split="source_val_clean",

                selected_epoch=state["epoch"],

                status="complete",

                mask_sha256=mask_digest.hexdigest(),

                achieved_removed_fraction=1 - visible / original,

                observed_cells=visible,

                original_observed_cells=original,

                recording_metrics=record_metrics,

                no_final_test_access=True,

                input_manifest_sha256=digest(args.data / "manifest.json"),

                selected_checkpoint_sha256=digest(args.training / "best.pth"),

            )

            for key, filename in [

                ("label_file", "labels.npy"),

                ("prediction_file", "predictions.npy"),

                ("logit_file", "logits.npy"),

            ]:

                result[key] = str(folder / filename)

                result[key + "_sha256"] = digest(folder / filename)

            save_json(done, result)

            records.append(result)

            save_json(

                args.output / "progress.json",

                {

                    "phase": "challenge_evaluation",

                    "completed_conditions": len(records),

                    "total_conditions": 2 * len(conditions()),

                    "split": role,

                    "condition": c["name"],

                    "time_unix": time.time(),

                },

            )

            print(

                json.dumps(

                    {

                        k: result[k]

                        for k in (

                            "model",

                            "split",

                            "condition",

                            "accuracy_percent",

                            "macro_f1_percent",

                        )

                    }

                ),

                flush=True,

            )

    save_json(

        args.output / "completion.json",

        {

            "status": "complete",

            "rows": records,

            "scope": "single_seed_challenge_development",

            "target_test_accessed": False,

        },

    )
