"""Raw Signal and CWT branches used by HAD-GCN."""

from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor, nn


class MaxNormConv2d(nn.Conv2d):
    def __init__(
        self,
        *args,
        max_norm: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.max_norm = float(max_norm)

    def forward(self, x: Tensor) -> Tensor:
        with torch.no_grad():
            self.weight.copy_(
                torch.renorm(
                    self.weight,
                    p=2,
                    dim=0,
                    maxnorm=self.max_norm,
                )
            )
        return super().forward(x)


class ChannelAttention1DLayout(nn.Module):
    """Attention for [B, channels, 1, time] tensors."""

    def __init__(
        self,
        channels: int,
        reduction: int = 8,
    ) -> None:
        super().__init__()
        hidden = max(1, channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.shared_mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        weights = self.shared_mlp(self.avg_pool(x))
        weights += self.shared_mlp(self.max_pool(x))
        return x * torch.sigmoid(weights)


class RawSignalFeatureExtractor(nn.Module):

    def __init__(self, n_chans: int = 22) -> None:
        super().__init__()
        if n_chans != 22:
            raise ValueError(
                "Raw Signal branch currently requires 22 EEG channels."
            )
        self.n_chans = n_chans

        self.main_branch = nn.Sequential(
            nn.Conv2d(1, 48, kernel_size=(11, 1), bias=False),
            nn.Conv2d(48, 64, kernel_size=(1, 25), bias=False),
            nn.Conv2d(64, 64, kernel_size=(12, 1), bias=False),
            nn.BatchNorm2d(64),
            nn.ELU(inplace=True),
        )
        self.auxiliary_branch = nn.Sequential(
            ChannelAttention1DLayout(n_chans),
            nn.Conv2d(
                n_chans,
                32,
                kernel_size=(1, 16),
                bias=False,
            ),
            MaxNormConv2d(
                32,
                64,
                kernel_size=(1, 25),
                groups=32,
                bias=False,
                max_norm=1.0,
            ),
            ChannelAttention1DLayout(64),
            nn.BatchNorm2d(64),
            nn.ELU(inplace=True),
            nn.AvgPool2d(
                kernel_size=(1, 180),
                stride=(1, 30),
            ),
        )

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim == 3:
            x = x.unsqueeze(1)
        if x.ndim != 4:
            raise ValueError(
                "Raw Signal branch expects [B, 1, C, T]."
            )
        if x.shape[1] != 1 or x.shape[2] != self.n_chans:
            raise ValueError(
                "Expected [B, 1, {}, T], got {}.".format(
                    self.n_chans,
                    tuple(x.shape),
                )
            )

        main_features = self.main_branch(x)
        auxiliary_features = self.auxiliary_branch(
            x.transpose(1, 2).contiguous()
        )

        if main_features.shape[:3] != auxiliary_features.shape[:3]:
            raise RuntimeError(
                "Raw branch outputs are incompatible: {} vs {}.".format(
                    tuple(main_features.shape),
                    tuple(auxiliary_features.shape),
                )
            )
        return torch.cat(
            (main_features, auxiliary_features),
            dim=-1,
        )


class RawSignalBranch(nn.Module):

    model_name = "Raw Signal branch"

    def __init__(
        self,
        n_chans: int = 22,
        n_times: int = 750,
        feature_dim: int = 128,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.n_chans = n_chans
        self.n_times = n_times
        self.feature_extractor = RawSignalFeatureExtractor(n_chans)
        self.fusion_pool = nn.AvgPool2d(
            kernel_size=(1, 180),
            stride=(1, 15),
        )
        flattened = self._infer_flattened_dim()
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened, feature_dim),
            nn.ELU(inplace=True),
            nn.Dropout(dropout),
        )

    def _infer_flattened_dim(self) -> int:
        was_training = self.feature_extractor.training
        self.feature_extractor.eval()
        try:
            with torch.no_grad():
                dummy = torch.zeros(
                    1,
                    1,
                    self.n_chans,
                    self.n_times,
                )
                features = self.fusion_pool(
                    self.feature_extractor(dummy)
                )
                return int(features.flatten(1).shape[1])
        finally:
            self.feature_extractor.train(was_training)

    def forward(self, x: Tensor) -> Tensor:
        features = self.feature_extractor(x)
        features = self.fusion_pool(features)
        return self.projection(features)


class ChannelAttention2D(nn.Module):
    def __init__(
        self,
        channels: int,
        reduction: int = 16,
    ) -> None:
        super().__init__()
        hidden = max(1, channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.shared_mlp = nn.Sequential(
            nn.Conv2d(
                channels,
                hidden,
                kernel_size=1,
                bias=False,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                hidden,
                channels,
                kernel_size=1,
                bias=False,
            ),
        )

    def forward(self, x: Tensor) -> Tensor:
        weights = self.shared_mlp(self.avg_pool(x))
        weights += self.shared_mlp(self.max_pool(x))
        return x * torch.sigmoid(weights)


class SpatialAttention2D(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            2,
            1,
            kernel_size=7,
            padding=3,
            bias=False,
        )

    def forward(self, x: Tensor) -> Tensor:
        average = x.mean(dim=1, keepdim=True)
        maximum = x.amax(dim=1, keepdim=True)
        weights = torch.sigmoid(
            self.conv(torch.cat((average, maximum), dim=1))
        )
        return x * weights


class ParallelCBAM(nn.Module):
    def __init__(
        self,
        channels: int,
        reduction: int = 16,
    ) -> None:
        super().__init__()
        self.channel_attention = ChannelAttention2D(
            channels,
            reduction,
        )
        self.spatial_attention = SpatialAttention2D()

    def forward(self, x: Tensor) -> Tensor:
        return (
            x
            + self.channel_attention(x)
            + self.spatial_attention(x)
        )


class ResidualDecompositionBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        bottleneck_channels: int,
        kernel_size: int = 5,
    ) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd.")
        padding = kernel_size // 2
        self.transform = nn.Sequential(
            nn.Conv2d(
                channels,
                bottleneck_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                bottleneck_channels,
                bottleneck_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(bottleneck_channels),
            nn.Conv2d(
                bottleneck_channels,
                channels,
                kernel_size=1,
                bias=False,
            ),
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.activation(x + self.transform(x))


class CWTBranch(nn.Module):

    model_name = "CWT branch"

    def __init__(
        self,
        input_channels: int = 1,
        feature_dim: int = 128,
        stem_channels: int = 128,
        stem_stride: int = 4,
        rdb_widths: Sequence[int] = (16, 32, 64),
        kernel_size: int = 5,
        attention_reduction: int = 16,
        pooled_size: int = 5,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if len(rdb_widths) != 3:
            raise ValueError("Exactly three RDB widths are required.")

        self.stem = nn.Sequential(
            nn.Conv2d(
                input_channels,
                stem_channels,
                kernel_size=1,
                stride=stem_stride,
                bias=False,
            ),
            nn.BatchNorm2d(stem_channels),
            nn.ReLU(inplace=True),
        )
        self.rdb1 = ResidualDecompositionBlock(
            stem_channels,
            rdb_widths[0],
            kernel_size,
        )
        self.rdb2 = ResidualDecompositionBlock(
            stem_channels,
            rdb_widths[1],
            kernel_size,
        )
        self.attention1 = ParallelCBAM(
            stem_channels,
            attention_reduction,
        )
        self.rdb3 = ResidualDecompositionBlock(
            stem_channels,
            rdb_widths[2],
            kernel_size,
        )
        self.attention2 = ParallelCBAM(
            stem_channels,
            attention_reduction,
        )
        self.dropout = nn.Dropout2d(dropout)
        self.pool = nn.AdaptiveAvgPool2d(
            (pooled_size, pooled_size)
        )
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                stem_channels * pooled_size * pooled_size,
                feature_dim,
            ),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward_feature_map(self, x: Tensor) -> Tensor:
        stem = self.stem(x)
        feature1 = self.rdb1(stem)
        feature2 = self.rdb2(feature1)
        fusion1 = self.attention1(feature1 + feature2)
        feature3 = self.rdb3(fusion1)
        return self.attention2(
            feature1 + fusion1 + feature3
        )

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4:
            raise ValueError(
                "CWT branch expects [B, 1, H, W]."
            )
        features = self.forward_feature_map(x)
        features = self.dropout(features)
        return self.projection(self.pool(features))
