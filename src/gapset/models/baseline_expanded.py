"""Seven new C1 adapters; pinned author classes and ordinary learnable weights."""

import ast
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

NAMES = (
    "sensefi_mlp",
    "sensefi_lenet",
    "sensefi_resnet18",
    "sensefi_lstm",
    "sensefi_vit",
    "amcs_tcn_c1",
    "dfs_csi_backbone",
)
YEARS = {
    **dict.fromkeys(NAMES[:5], 2023),
    "amcs_tcn_c1": 2026,
    "dfs_csi_backbone": 2026,
    "sensefi_cnn_gru": 2023,
    "wiflexformer": 2024,
    "csibert2_clean_backbone": 2025,
    "hms_rwkv_full": 2026,
}
OLD_NAMES = ("sensefi_cnn_gru", "wiflexformer", "csibert2_clean_backbone", "hms_rwkv_full")

def author_classes(path, selected, *, einops=False):
    """Compile exact author class/function ASTs without top-level train/data imports."""
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    nodes = [
        n
        for n in tree.body
        if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in selected
    ]
    if {n.name for n in nodes} != set(selected):
        raise ValueError("Pinned author definitions missing")
    namespace = {"torch": torch, "nn": nn, "F": F, "__name__": "pinned_expanded_author"}
    if einops:
        from einops import rearrange, reduce, repeat
        from einops.layers.torch import Rearrange, Reduce

        namespace.update(
            rearrange=rearrange, reduce=reduce, repeat=repeat, Rearrange=Rearrange, Reduce=Reduce
        )
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)  # noqa: S102
    return namespace

class SenseFiMLP(nn.Module):
    def __init__(self, vendor, frequencies, steps, classes):
        super().__init__()
        author = author_classes(vendor / "sensefi/widar_model.py", ["Widar_MLP"])
        self.core = author["Widar_MLP"](classes)
        self.core.fc[0] = nn.Linear(8 * frequencies * steps, 1024)

    def forward(self, image, timestamp):
        return self.core.fc(image.flatten(1))

class SenseFiLeNet(nn.Module):
    def __init__(self, vendor, frequencies, steps, classes):
        super().__init__()
        author = author_classes(vendor / "sensefi/widar_model.py", ["Widar_LeNet"])
        self.core = author["Widar_LeNet"](classes)
        self.core.encoder[0] = nn.Conv2d(8, 32, 6, stride=2)
        height, width = [(value - 6) // 2 + 1 - 4 for value in (frequencies, steps)]
        if min(height, width) <= 0:
            raise ValueError("Native C1 axes too short for the author LeNet kernels")
        self.core.fc[0] = nn.Linear(96 * height * width, 128)

    def forward(self, image, timestamp):
        return self.core.fc(self.core.encoder(image).flatten(1))

class SenseFiResNet(nn.Module):
    def __init__(self, vendor, classes):
        super().__init__()
        author = author_classes(
            vendor / "sensefi/widar_model.py", ["Block", "Widar_ResNet", "Widar_ResNet18"]
        )
        self.core = author["Widar_ResNet18"](classes)
        self.core.reshape[0] = nn.ConvTranspose2d(8, 3, 7, stride=1)

    def forward(self, image, timestamp):
        return self.core(image)

class SenseFiLSTM(nn.Module):
    def __init__(self, vendor, frequencies, classes):
        super().__init__()
        author = author_classes(vendor / "sensefi/widar_model.py", ["Widar_LSTM"])
        self.core = author["Widar_LSTM"](classes)
        self.core.lstm = nn.LSTM(8 * frequencies, 64, num_layers=1)

    def forward(self, image, timestamp):
        sequence = image.permute(3, 0, 1, 2).flatten(2)
        _, (hidden, _) = self.core.lstm(sequence)
        return self.core.fc(hidden[-1])

class SenseFiViT(nn.Module):
    def __init__(self, vendor, frequencies, steps, classes):
        super().__init__()
        selected = [
            "PatchEmbedding",
            "MultiHeadAttention",
            "ResidualAdd",
            "FeedForwardBlock",
            "TransformerEncoderBlock",
            "TransformerEncoder",
            "ClassificationHead",
            "Widar_ViT",
        ]
        author = author_classes(vendor / "sensefi/widar_model.py", selected, einops=True)
        self.pad_f = (-frequencies) % 2
        self.pad_t = (-steps) % 40
        tokens = math.ceil(frequencies / 2) * math.ceil(steps / 40)
        self.core = author["Widar_ViT"](
            in_channels=8,
            patch_size_w=2,
            patch_size_h=40,
            emb_size=80,
            img_size=tokens * 80,
            depth=1,
            num_classes=classes,
        )

    def forward(self, image, timestamp):
        # Hardcoded BVP view is bypassed; exact source patch/CLS/position/attention/readout remain.
        x = self.core[0].projection(F.pad(image, (0, self.pad_t, 0, self.pad_f)))
        cls = self.core[0].cls_token.expand(len(image), -1, -1)
        x = torch.cat((cls, x), dim=1) + self.core[0].position
        return self.core[2](self.core[1](x))

class AMCS(nn.Module):
    def __init__(self, vendor, frequencies, classes):
        super().__init__()
        selected = [
            "MCS_FMFA_Block",
            "FCA_Block",
            "DepthwiseSeparableConv1d",
            "TCN_ResBlock",
            "TCN_Backbone",
            "AMCS_TCN",
        ]
        author = author_classes(vendor / "amcs/amcs_tcn.py", selected)
        # Shared pointwise projection per antenna, no mixing or synthesizing antenna links.
        self.stem = nn.Linear(4, 1)
        self.core = author["AMCS_TCN"](
            num_classes=classes, cnn_channels=32, subcarriers=frequencies
        )

    def public_fields(self, image):
        amplitude = image[:, :3]
        mask = image[:, 3:6]
        time = image[:, 6:7].expand_as(amplitude)
        ordinal = image[:, 7:8].expand_as(amplitude)
        return torch.stack((amplitude, mask, time, ordinal), dim=-1)

    def forward(self, image, timestamp):
        x = self.stem(self.public_fields(image)).squeeze(-1).transpose(2, 3)
        return self.core(x)

class DFS(nn.Module):
    def __init__(self, vendor, frequencies, classes):
        super().__init__()
        author = author_classes(
            vendor / "dfs/test4amplitude48canais.py", ["PositionalEncoding", "TransformerModel"]
        )
        self.stem = nn.Linear(8 * frequencies, 48)
        # Public Run specialist constructor: d48, four heads, two layers, dropout0.1.
        self.core = author["TransformerModel"](
            input_size=48, output_size=classes, num_heads=4, num_layers=2
        )

    def forward(self, image, timestamp):
        sequence = image.permute(0, 3, 1, 2).flatten(2)
        return self.core(self.stem(sequence))

def make_model(name, vendor, frequencies=30, steps=256, classes=6):
    vendor = Path(vendor)
    factories = {
        "sensefi_mlp": lambda: SenseFiMLP(vendor, frequencies, steps, classes),
        "sensefi_lenet": lambda: SenseFiLeNet(vendor, frequencies, steps, classes),
        "sensefi_resnet18": lambda: SenseFiResNet(vendor, classes),
        "sensefi_lstm": lambda: SenseFiLSTM(vendor, frequencies, classes),
        "sensefi_vit": lambda: SenseFiViT(vendor, frequencies, steps, classes),
        "amcs_tcn_c1": lambda: AMCS(vendor, frequencies, classes),
        "dfs_csi_backbone": lambda: DFS(vendor, frequencies, classes),
    }
    if name not in factories:
        raise ValueError("Only the seven explicitly new cases may train: " + name)
    return factories[name]()

def core_paths(name):
    return {
        "sensefi_mlp": ("core.fc",),
        "sensefi_lenet": ("core.encoder.0", "core.encoder.2", "core.encoder.4", "core.fc"),
        "sensefi_resnet18": (
            "core.reshape",
            "core.conv1",
            "core.layer1",
            "core.layer2",
            "core.layer3",
            "core.layer4",
            "core.fc",
        ),
        "sensefi_lstm": ("core.lstm", "core.fc"),
        "sensefi_vit": ("core.0.projection", "core.1", "core.2"),
        "amcs_tcn_c1": (
            "stem",
            "core.antenna_streams.0",
            "core.antenna_streams.1",
            "core.antenna_streams.2",
            "core.fca_fusion",
            "core.tcn.network.0",
            "core.tcn.network.1",
            "core.tcn.network.2",
            "core.tcn.network.3",
            "core.classifier",
        ),
        "dfs_csi_backbone": (
            "stem",
            "core.pos_encoder",
            "core.transformer_encoder.layers.0",
            "core.transformer_encoder.layers.1",
            "core.fc",
        ),
    }[name]
