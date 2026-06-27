"""Configuration for the HAD-GCN reproduction."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple


# ============================================================================
# User settings
# ============================================================================
SUBJECT_ID = 2
DATA_FOLDER = Path(r"D:\EEG\BCI2a")
OUTPUT_FOLDER = Path("outputs")
CACHE_FOLDER = Path("cache")


@dataclass
class ExperimentConfig:


    subject_id: int = SUBJECT_ID
    data_folder: Path = field(default_factory=lambda: DATA_FOLDER)
    output_folder: Path = field(default_factory=lambda: OUTPUT_FOLDER)
    cache_folder: Path = field(default_factory=lambda: CACHE_FOLDER)


    protocol: str = "subject_specific"
    source_subjects: Optional[Tuple[int, ...]] = None
    validation_fraction: float = 0.2


    preprocessing_band_hz: Tuple[float, float] = (0.5, 32.0)
    trial_window_s: Tuple[float, float] = (3.0, 6.0)
    expected_sampling_rate_hz: float = 250.0
    filter_order: int = 4

    # Graph-learning feature extraction.
    graph_subbands_hz: Tuple[Tuple[float, float], ...] = (
        (8.0, 12.0),
        (12.0, 16.0),
        (16.0, 20.0),
        (20.0, 24.0),
        (24.0, 28.0),
        (28.0, 32.0),
    )
    graph_features_per_band: int = 2
    ecg_placeholder_dim: int = 3
    pli_top_k: int = 6
    pli_min_weight: float = 0.0

    cwt_selected_channels: Tuple[str, str, str] = (
        "EEG-C3",
        "EEG-Cz",
        "EEG-C4",
    )
    cwt_frequency_hz: Tuple[float, float] = (8.0, 32.0)
    cwt_frequency_bins: int = 64
    cwt_morlet_cycles: float = 7.0
    cwt_image_size: int = 160
    cwt_workers: int = 0
    rebuild_cache: bool = False

    # This manual value is only a runnable placeholder: "raw" or "cwt".
    manual_branch_placeholder: str = "cwt"


    raw_branch_feature_dim: int = 128
    cwt_branch_feature_dim: int = 128
    graph_hidden_dims: Tuple[int, int, int] = (64, 64, 128)
    graph_feature_dim: int = 128
    fusion_hidden_dim: int = 128
    n_classes: int = 4
    dropout: float = 0.3


    cwt_stem_channels: int = 128
    cwt_stem_stride: int = 4
    cwt_rdb_widths: Tuple[int, int, int] = (16, 32, 64)
    cwt_kernel_size: int = 5
    cwt_attention_reduction: int = 16
    cwt_pooled_size: int = 5

    epochs: int = 700
    batch_size: int = 64
    learning_rate: float = 0.003
    weight_decay: float = 0.0
    early_stopping_enabled: bool = True
    patience: int = 20
    seed: int = 216
    num_workers: int = 0
    log_period: int = 1

    # Trial-graph confidence calibration.
    calibration_enabled: bool = True
    calibration_epochs: int = 200
    calibration_learning_rate: float = 0.01
    calibration_hidden_dim: int = 16
    calibration_top_k: int = 10
    confidence_threshold: float = 0.0

    def validate(self) -> None:
        if not 1 <= self.subject_id <= 9:
            raise ValueError("subject_id must be between 1 and 9.")
        if self.protocol not in {"subject_specific", "cross_subject"}:
            raise ValueError("protocol must be subject_specific or cross_subject.")
        if not 0.0 < self.validation_fraction < 1.0:
            raise ValueError("validation_fraction must be between 0 and 1.")
        if self.manual_branch_placeholder not in {"raw", "cwt"}:
            raise ValueError("manual_branch_placeholder must be raw or cwt.")
        if self.graph_features_per_band != 2:
            raise ValueError(
                "This implementation uses two public graph features per band."
            )
        if len(self.graph_hidden_dims) != 3:
            raise ValueError("HAD-GCN uses exactly three GCN layers.")
        if self.raw_branch_feature_dim != self.cwt_branch_feature_dim:
            raise ValueError(
                "Raw and CWT branch embeddings must have the same dimension."
            )
        if self.epochs <= 0 or self.batch_size <= 0:
            raise ValueError("epochs and batch_size must be positive.")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if self.patience <= 0:
            raise ValueError("patience must be positive.")
        if self.cwt_image_size <= 0 or self.cwt_frequency_bins <= 1:
            raise ValueError("Invalid CWT image/frequency settings.")
        if self.pli_top_k < 0 or self.calibration_top_k < 0:
            raise ValueError("top-k values cannot be negative.")
