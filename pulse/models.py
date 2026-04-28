from __future__ import annotations

from typing import Dict, Iterable, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .specs import TaskSpec, TaskType


class ConvNormAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DepthwiseSeparableBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=in_channels,
            bias=False,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)
        self.use_residual = stride == 1 and in_channels == out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        if self.use_residual:
            x = x + identity
        return self.act(x)


class TinyEncoder(nn.Module):
    def __init__(self, in_channels: int = 3, width: int = 24):
        super().__init__()
        self.stem = ConvNormAct(in_channels, width, kernel_size=3, stride=2)
        self.stage1 = nn.Sequential(
            DepthwiseSeparableBlock(width, width),
            DepthwiseSeparableBlock(width, width * 2, stride=2),
        )
        self.stage2 = nn.Sequential(
            DepthwiseSeparableBlock(width * 2, width * 2),
            DepthwiseSeparableBlock(width * 2, width * 3, stride=2),
        )
        self.stage3 = nn.Sequential(
            DepthwiseSeparableBlock(width * 3, width * 3),
            DepthwiseSeparableBlock(width * 3, width * 5, stride=2),
        )
        self.stage4 = nn.Sequential(
            DepthwiseSeparableBlock(width * 5, width * 5),
            DepthwiseSeparableBlock(width * 5, width * 8, stride=2),
        )
        self.embedding_dim = width * 8

    def forward_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.stem(x)
        f1 = self.stage1(x)
        f2 = self.stage2(f1)
        f3 = self.stage3(f2)
        f4 = self.stage4(f3)
        return [f1, f2, f3, f4]

    def forward_embedding(self, x: torch.Tensor) -> torch.Tensor:
        features = self.forward_features(x)
        return F.adaptive_avg_pool2d(features[-1], 1).flatten(1)


class ClassificationModel(nn.Module):
    def __init__(self, num_classes: int, encoder_width: int = 24, hidden_dim: int = 160, dropout: float = 0.2):
        super().__init__()
        self.encoder = TinyEncoder(width=encoder_width)
        self.head = nn.Sequential(
            nn.Linear(self.encoder.embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder.forward_embedding(x))


class RegressionModel(nn.Module):
    def __init__(self, encoder_width: int = 24, hidden_dim: int = 160, dropout: float = 0.15):
        super().__init__()
        self.encoder = TinyEncoder(width=encoder_width)
        self.head = nn.Sequential(
            nn.Linear(self.encoder.embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder.forward_embedding(x)).squeeze(1)


class DetectionModel(nn.Module):
    def __init__(self, encoder_width: int = 24, hidden_dim: int = 192, dropout: float = 0.2):
        super().__init__()
        self.encoder = TinyEncoder(width=encoder_width)
        self.head = nn.Sequential(
            nn.Linear(self.encoder.embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 5),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        raw = self.head(self.encoder.forward_embedding(x))
        return {
            "objectness": raw[:, :1],
            "bbox": raw[:, 1:].sigmoid(),
        }


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = ConvNormAct(in_channels + skip_channels, out_channels, kernel_size=3)
        self.conv2 = ConvNormAct(out_channels, out_channels, kernel_size=3)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class SegmentationModel(nn.Module):
    def __init__(self, out_channels: int, encoder_width: int = 24):
        super().__init__()
        self.encoder = TinyEncoder(width=encoder_width)
        width = encoder_width
        self.up3 = UpBlock(width * 8, width * 5, width * 5)
        self.up2 = UpBlock(width * 5, width * 3, width * 3)
        self.up1 = UpBlock(width * 3, width * 2, width * 2)
        self.up0 = UpBlock(width * 2, width * 2, width * 2)
        self.head = nn.Conv2d(width * 2, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[-2:]
        f1, f2, f3, f4 = self.encoder.forward_features(x)
        x = self.up3(f4, f3)
        x = self.up2(x, f2)
        x = self.up1(x, f1)
        x = self.up0(x, f1)
        logits = self.head(x)
        return F.interpolate(logits, size=input_size, mode="bilinear", align_corners=False)


class MultiModalFusionModel(nn.Module):
    def __init__(self, modalities: Iterable[str], num_classes: int, encoder_width: int = 24, hidden_dim: int = 160, dropout: float = 0.2):
        super().__init__()
        self.modalities = list(modalities)
        self.encoders = nn.ModuleDict({name: TinyEncoder(width=encoder_width) for name in self.modalities})
        fusion_width = len(self.modalities) * self.encoders[self.modalities[0]].embedding_dim
        self.gate = nn.Sequential(
            nn.Linear(fusion_width, 128),
            nn.GELU(),
            nn.Linear(128, len(self.modalities)),
        )
        self.head = nn.Sequential(
            nn.Linear(self.encoders[self.modalities[0]].embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, modalities: Dict[str, torch.Tensor]) -> torch.Tensor:
        embeddings = [self.encoders[name].forward_embedding(modalities[name]) for name in self.modalities]
        stacked = torch.cat(embeddings, dim=1)
        weights = torch.softmax(self.gate(stacked), dim=1)
        fused = sum(weights[:, idx:idx + 1] * emb for idx, emb in enumerate(embeddings))
        return self.head(fused)


def build_model(task: TaskSpec) -> nn.Module:
    encoder_width = int(task.extras.get("encoder_width", 24))
    hidden_dim = int(task.extras.get("hidden_dim", 160))
    dropout = float(task.extras.get("dropout", 0.2))
    if task.task_type in {TaskType.CLASSIFICATION}:
        return ClassificationModel(
            num_classes=len(task.labels),
            encoder_width=encoder_width,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
    if task.task_type == TaskType.MULTIMODAL:
        return MultiModalFusionModel(
            task.extras.get("modalities", []),
            num_classes=len(task.labels),
            encoder_width=encoder_width,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
    if task.task_type == TaskType.SEGMENTATION:
        if task.extras.get("segmentation_mode", "binary") == "binary":
            out_channels = 1
        else:
            out_channels = len(task.extras.get("mask_class_names", []))
        return SegmentationModel(out_channels=out_channels, encoder_width=encoder_width)
    if task.task_type == TaskType.DETECTION:
        return DetectionModel(encoder_width=encoder_width, hidden_dim=max(hidden_dim, 192), dropout=dropout)
    return RegressionModel(encoder_width=encoder_width, hidden_dim=hidden_dim, dropout=min(dropout, 0.15))
