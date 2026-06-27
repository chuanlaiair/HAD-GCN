"""HAD-GCN model with patent-sensitive components represented by variables."""

from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import Tensor, nn

try:
    from .branches import CWTBranch, RawSignalBranch
    from .graph_model import GraphLearningModule
    from .proprietary_placeholders import (
        resolve_ecg_adaptive_features,
        resolve_tf_branch_selection,
    )
except ImportError:
    from branches import CWTBranch, RawSignalBranch
    from graph_model import GraphLearningModule
    from proprietary_placeholders import (
        resolve_ecg_adaptive_features,
        resolve_tf_branch_selection,
    )


class HADGCN(nn.Module):
    """Hybrid Adaptive Domain Graph Convolutional Network.

    The model contains all public components. The adaptive selector and ECG
    generator are NOT implemented; their outputs are accepted as variables.
    """

    model_name = "HAD-GCN"

    def __init__(
        self,
        n_chans: int,
        n_times: int,
        eeg_node_feature_dim: int,
        ecg_placeholder_dim: int,
        raw_branch_feature_dim: int,
        cwt_branch_feature_dim: int,
        graph_hidden_dims,
        graph_feature_dim: int,
        fusion_hidden_dim: int,
        n_classes: int,
        dropout: float,
        manual_branch_placeholder: str,
        cwt_stem_channels: int,
        cwt_stem_stride: int,
        cwt_rdb_widths,
        cwt_kernel_size: int,
        cwt_attention_reduction: int,
        cwt_pooled_size: int,
    ) -> None:
        super().__init__()
        if raw_branch_feature_dim != cwt_branch_feature_dim:
            raise ValueError(
                "The two time-frequency branches must output equal dimensions."
            )

        self.ecg_placeholder_dim = ecg_placeholder_dim
        self.manual_branch_placeholder = manual_branch_placeholder
        self.time_frequency_feature_dim = raw_branch_feature_dim

        self.raw_signal_branch = RawSignalBranch(
            n_chans=n_chans,
            n_times=n_times,
            feature_dim=raw_branch_feature_dim,
            dropout=dropout,
        )
        self.cwt_branch = CWTBranch(
            input_channels=1,
            feature_dim=cwt_branch_feature_dim,
            stem_channels=cwt_stem_channels,
            stem_stride=cwt_stem_stride,
            rdb_widths=cwt_rdb_widths,
            kernel_size=cwt_kernel_size,
            attention_reduction=cwt_attention_reduction,
            pooled_size=cwt_pooled_size,
            dropout=dropout,
        )
        self.graph_learning_module = GraphLearningModule(
            input_dim=eeg_node_feature_dim + ecg_placeholder_dim,
            hidden_dims=graph_hidden_dims,
            output_dim=graph_feature_dim,
            dropout=dropout,
        )
        self.feature_fusion = nn.Sequential(
            nn.Linear(
                graph_feature_dim + raw_branch_feature_dim,
                fusion_hidden_dim,
            ),
            nn.BatchNorm1d(fusion_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(
            fusion_hidden_dim,
            n_classes,
        )

    def _extract_selected_branch(
        self,
        raw_eeg: Tensor,
        cwt_image: Tensor,
        branch_selection: Tensor,
    ) -> Tensor:
        batch_size = raw_eeg.shape[0]
        output = torch.zeros(
            batch_size,
            self.time_frequency_feature_dim,
            device=raw_eeg.device,
            dtype=raw_eeg.dtype,
        )

        raw_mask = branch_selection == 0
        cwt_mask = branch_selection == 1

        if torch.any(raw_mask):
            output[raw_mask] = self.raw_signal_branch(
                raw_eeg[raw_mask]
            )
        if torch.any(cwt_mask):
            output[cwt_mask] = self.cwt_branch(
                cwt_image[cwt_mask]
            )
        return output

    def forward(
        self,
        raw_eeg: Tensor,
        cwt_image: Tensor,
        graph_node_features: Tensor,
        graph_adjacency: Tensor,
        tf_branch_selection: Optional[Tensor] = None,
        ecg_adaptive_features: Optional[Tensor] = None,
        return_features: bool = False,
    ):
        batch_size, node_count, _ = graph_node_features.shape

        branch_selection = resolve_tf_branch_selection(
            batch_size=batch_size,
            device=raw_eeg.device,
            explicit_selection=tf_branch_selection,
            manual_branch=self.manual_branch_placeholder,
        )
        ecg_features = resolve_ecg_adaptive_features(
            batch_size=batch_size,
            node_count=node_count,
            feature_dim=self.ecg_placeholder_dim,
            device=graph_node_features.device,
            dtype=graph_node_features.dtype,
            explicit_features=ecg_adaptive_features,
        )

        complete_node_features = torch.cat(
            (graph_node_features, ecg_features),
            dim=-1,
        )
        graph_feature, gcn_layers = self.graph_learning_module(
            complete_node_features,
            graph_adjacency,
        )
        time_frequency_feature = self._extract_selected_branch(
            raw_eeg,
            cwt_image,
            branch_selection,
        )
        fused_feature = self.feature_fusion(
            torch.cat(
                (graph_feature, time_frequency_feature),
                dim=-1,
            )
        )
        logits = self.classifier(fused_feature)

        if not return_features:
            return logits

        return {
            "logits": logits,
            "graph_feature": graph_feature,
            "time_frequency_feature": time_frequency_feature,
            "fused_feature": fused_feature,
            "branch_selection": branch_selection,
            "ecg_adaptive_features": ecg_features,
            "gcn_layer_1": gcn_layers[0],
            "gcn_layer_2": gcn_layers[1],
            "gcn_layer_3": gcn_layers[2],
        }
