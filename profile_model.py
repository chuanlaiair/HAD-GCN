"""Report actual parameter counts for the implemented model."""

from __future__ import annotations

import torch

try:
    from .config import ExperimentConfig
    from .had_gcn import HADGCN
    from .utils import (
        count_trainable_parameters,
        module_parameter_counts,
    )
except ImportError:
    from config import ExperimentConfig
    from had_gcn import HADGCN
    from utils import (
        count_trainable_parameters,
        module_parameter_counts,
    )


def main() -> None:
    config = ExperimentConfig()
    raw_time_points = int(
        (
            config.trial_window_s[1]
            - config.trial_window_s[0]
        )
        * config.expected_sampling_rate_hz
    )
    eeg_node_feature_dim = (
        len(config.graph_subbands_hz)
        * config.graph_features_per_band
    )

    model = HADGCN(
        n_chans=22,
        n_times=raw_time_points,
        eeg_node_feature_dim=eeg_node_feature_dim,
        ecg_placeholder_dim=config.ecg_placeholder_dim,
        raw_branch_feature_dim=config.raw_branch_feature_dim,
        cwt_branch_feature_dim=config.cwt_branch_feature_dim,
        graph_hidden_dims=config.graph_hidden_dims,
        graph_feature_dim=config.graph_feature_dim,
        fusion_hidden_dim=config.fusion_hidden_dim,
        n_classes=config.n_classes,
        dropout=config.dropout,
        manual_branch_placeholder=config.manual_branch_placeholder,
        cwt_stem_channels=config.cwt_stem_channels,
        cwt_stem_stride=config.cwt_stem_stride,
        cwt_rdb_widths=config.cwt_rdb_widths,
        cwt_kernel_size=config.cwt_kernel_size,
        cwt_attention_reduction=config.cwt_attention_reduction,
        cwt_pooled_size=config.cwt_pooled_size,
    )

    print("Actual trainable parameters:", count_trainable_parameters(model))
    for name, count in module_parameter_counts(model).items():
        print("{:<28s} {}".format(name, count))

    batch_size = 2
    with torch.no_grad():
        output = model(
            raw_eeg=torch.randn(
                batch_size,
                1,
                22,
                raw_time_points,
            ),
            cwt_image=torch.randn(
                batch_size,
                1,
                config.cwt_image_size,
                config.cwt_image_size,
            ),
            graph_node_features=torch.randn(
                batch_size,
                22,
                eeg_node_feature_dim,
            ),
            graph_adjacency=torch.eye(22).repeat(
                batch_size,
                1,
                1,
            ),
        )
    print("Output shape:", tuple(output.shape))


if __name__ == "__main__":
    main()
