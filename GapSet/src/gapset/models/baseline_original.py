"""Thin C1 wrappers around pinned author core classes, not trained checkpoints."""

import ast
import importlib.util
import sys
from pathlib import Path

import torch
from torch import nn

NAMES = ("sensefi_cnn_gru", "wiflexformer", "csibert2_clean_backbone", "hms_rwkv_full")
YEARS = dict(zip(NAMES, (2023, 2024, 2025, 2026), strict=True))

def module_file(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

def sensefi_class(path):
    """Execute the exact selected author class AST without unused torchvision.

    No model-class edits: imports of unused other architectures are excluded.
    Wrappers call encoder/mean/GRU/dropout/Linear directly to replace hardcoded
    ancestral view sizes and omit probability Softmax before common CE.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    nodes = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "NTU_Fi_CNN_GRU"
    ]
    if len(nodes) != 1:
        raise ValueError("Author CNN-GRU class not unique")
    selected = ast.Module(body=nodes, type_ignores=[])
    namespace = {"torch": torch, "nn": nn, "__name__": "pinned_sensefi_selected_class"}
    exec(compile(selected, str(path), "exec"), namespace)  # noqa: S102 - exact pinned author class AST
    return namespace["NTU_Fi_CNN_GRU"]

class SenseFi(nn.Module):
    def __init__(self, vendor, frequencies, classes):
        super().__init__()
        self.core = sensefi_class(vendor / "sensefi/NTU_Fi_model.py")(classes)
        self.stem = nn.Linear(8 * frequencies, 342)

    def forward(self, image, timestamp):
        batch, _, _, steps = image.shape
        x = self.stem(image.permute(0, 3, 1, 2).flatten(2))
        x = self.core.encoder(x.reshape(batch * steps, 1, 342))
        x = self.core.mean(x.permute(0, 2, 1)).reshape(batch, steps, 8)
        _, hidden = self.core.gru(x.transpose(0, 1))
        return self.core.classifier[1](self.core.classifier[0](hidden[-1]))

class WiFlex(nn.Module):
    def __init__(self, vendor, frequencies, steps, classes):
        super().__init__()
        author = module_file(vendor / "wiflexformer/wiflexformer.py", "pinned_common_wiflex")
        self.core = author.WiFlexFormer(
            input_dim=frequencies,
            num_channels=8,
            feature_dim=32,
            num_heads=16,
            num_layers=4,
            dim_feedforward=64,
            window_size=steps,
            num_classes=classes,
        )

    def forward(self, image, timestamp):
        return self.core(image)

class CSIBert2(nn.Module):
    def __init__(self, vendor, frequencies, steps, classes):
        super().__init__()
        from transformers import BertConfig

        author = module_file(vendor / "csibert2/model.py", "pinned_common_csibert2")
        configuration = BertConfig(
            max_position_embeddings=steps,
            hidden_size=128,
            num_hidden_layers=6,
            num_attention_heads=8,
            intermediate_size=512,
        )
        self.core = author.Sequence_Classifier(
            author.CSIBERT(configuration, 8 * frequencies), classes, hs=128, da=128, r=4
        )
        # Only terminal classifier is standardized. Author attention readout stays.
        self.core.classifier = nn.Linear(128 * 4, classes)

    def forward(self, image, timestamp):
        values = image.permute(0, 3, 1, 2).flatten(2)
        # Padding/entire-step observability are already in the public mask.
        support = image[:, 3:6].any(dim=(1, 2))
        return self.core(values, timestamp, attention_mask=support)

class HMS(nn.Module):
    def __init__(self, vendor, frequencies, classes):
        super().__init__()
        package = vendor / "hms"
        spec = importlib.util.spec_from_file_location(
            "pinned_common_hms", package / "__init__.py", submodule_search_locations=[str(package)]
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        author = __import__("pinned_common_hms.hms_rwkv", fromlist=["HMSRWKV"])
        self.core = author.HMSRWKV(
            in_channels=8,
            rwkv_input_dim=8 * frequencies,
            num_classes=classes,
            dataset="Widar",
            hidden_dim=36,
            num_layers=2,
            num_heads=6,
            dropout=0.15,
        )

    def forward(self, image, timestamp):
        temporal = image.permute(0, 3, 1, 2).flatten(2)
        return self.core({"rwkv_data": temporal, "spatial_data": image})["logits"]

def make_model(name, vendor, frequencies=30, steps=256, classes=6):
    vendor = Path(vendor)
    if name == "sensefi_cnn_gru":
        return SenseFi(vendor, frequencies, classes)
    if name == "wiflexformer":
        return WiFlex(vendor, frequencies, steps, classes)
    if name == "csibert2_clean_backbone":
        return CSIBert2(vendor, frequencies, steps, classes)
    if name == "hms_rwkv_full":
        return HMS(vendor, frequencies, classes)
    raise ValueError("Unfrozen model: " + name)

def core_paths(name):
    return {
        "sensefi_cnn_gru": (
            "core.encoder",
            "core.mean",
            "core.gru",
            "core.classifier.0",
            "core.classifier.1",
        ),
        "wiflexformer": (
            "core.stem2D",
            "core.stem1D",
            "core.pos_encoding",
            "core.transformer_encoder",
            "core.class_token_output",
        ),
        "csibert2_clean_backbone": (
            "core.bert.arl",
            "core.bert.csi_emb",
            "core.bert.time_emb",
            "core.bert.bert",
            "core.attention",
            "core.classifier",
        ),
        "hms_rwkv_full": (
            "core.input_projection",
            "core.rwkv_blocks.0",
            "core.rwkv_blocks.1",
            "core.spatial_branch.layer1",
            "core.spatial_branch.simam1",
            "core.spatial_branch.layer2",
            "core.spatial_branch.simam2",
            "core.spatial_branch.layer3",
            "core.agf",
            "core.classifier",
        ),
    }[name]
