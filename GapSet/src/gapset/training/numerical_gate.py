"""Actual CUDA selective-scan vs literal CPU recurrence, before isolated update."""

import torch
from gapset.data.dataset import C1Dataset
from gapset.models.registry import make_model

from gapset.data.preprocess import save_json


def check(args, model, guard):
    ds = C1Dataset(args.data, "fit_clean")
    examples = [ds[0], ds[len(ds) - 1]]
    image = torch.stack([r[0] for r in examples])
    ts = torch.stack([r[1] for r in examples])
    labels = torch.tensor([r[2] for r in examples])
    reference = make_model(args.model, args.vendor).eval()
    reference.load_state_dict(
        {k: v.detach().cpu() for k, v in model.state_dict().items()}, strict=True
    )
    model.eval()
    model.zero_grad(set_to_none=True)
    predicted = reference(image, ts)
    loss = torch.nn.functional.cross_entropy(predicted, labels)
    loss.backward()
    actual = model(image.cuda(), ts.cuda())
    torch.nn.functional.cross_entropy(actual, labels.cuda()).backward()
    torch.cuda.synchronize()
    error = (actual.detach().cpu() - predicted.detach()).abs().max().item()
    max_grad = 0.0
    sum_diff = 0.0
    sum_ref = 0.0
    cpu = dict(reference.named_parameters())
    for name, p in model.named_parameters():
        if p.grad is None and cpu[name].grad is None and args.model in ("csibert2_clean_backbone", "dfs_csi_backbone"):
            continue
        if p.grad is None or cpu[name].grad is None:
            raise ValueError("Missing paired-scan gradient")
        ref = cpu[name].grad
        diff = p.grad.detach().cpu() - ref
        max_grad = max(max_grad, diff.abs().max().item())
        sum_diff += diff.square().sum().item()
        sum_ref += ref.square().sum().item()
    relative = (sum_diff / max(sum_ref, 1e-20)) ** 0.5
    if error > 2e-4 or max_grad > 3e-4 or relative > 5e-3:
        raise ValueError(f"Fused/CPU recurrence parity failed:{error},{max_grad},{relative}")
    model.zero_grad(set_to_none=True)
    guard.check(True)
    save_json(
        args.output / "fused_cpu_parity.json",
        {
            "status": "passed",
            "real_native_input": True,
            "fused_required": args.model.startswith("ours_"),
            "logit_maxdiff": error,
            "gradient_maxdiff": max_grad,
            "gradient_relative_l2": relative,
            "optimizer_step": False,
        },
    )
