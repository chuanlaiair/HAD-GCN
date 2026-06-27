"""HAD-GCN experiment orchestration."""

from __future__ import annotations

import logging
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import nn

try:
    from .calibration import (
        apply_calibrator,
        calibration_metrics,
        train_calibrator,
    )
    from .config import ExperimentConfig
    from .data_pipeline import (
        HADGCNDataset,
        PreparedSession,
        concatenate_sessions,
        make_loader,
        prepare_session,
    )
    from .had_gcn import HADGCN
    from .graph_features import FBCSPNodeFeatureExtractor
    from .training import (
        collect_outputs,
        evaluate,
        fit,
        save_history,
    )
    from .utils import (
        count_trainable_parameters,
        module_parameter_counts,
        resolve_device,
        save_json,
        set_random_seed,
    )
except ImportError:
    from calibration import (
        apply_calibrator,
        calibration_metrics,
        train_calibrator,
    )
    from config import ExperimentConfig
    from data_pipeline import (
        HADGCNDataset,
        PreparedSession,
        concatenate_sessions,
        make_loader,
        prepare_session,
    )
    from had_gcn import HADGCN
    from graph_features import FBCSPNodeFeatureExtractor
    from training import (
        collect_outputs,
        evaluate,
        fit,
        save_history,
    )
    from utils import (
        count_trainable_parameters,
        module_parameter_counts,
        resolve_device,
        save_json,
        set_random_seed,
    )

LOGGER = logging.getLogger(__name__)


def _split_session(
    session: PreparedSession,
    validation_fraction: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    indices = np.arange(len(session.labels))
    train_indices, validation_indices = train_test_split(
        indices,
        test_size=validation_fraction,
        random_state=seed,
        shuffle=True,
        stratify=session.labels,
    )
    return train_indices, validation_indices


def _prepare_protocol(
    config: ExperimentConfig,
) -> Tuple[
    PreparedSession,
    np.ndarray,
    np.ndarray,
    PreparedSession,
]:
    if config.protocol == "subject_specific":
        training_session = prepare_session(
            config,
            config.subject_id,
            "T",
        )
        test_session = prepare_session(
            config,
            config.subject_id,
            "E",
        )
    else:
        source_subjects = (
            config.source_subjects
            if config.source_subjects is not None
            else tuple(
                subject
                for subject in range(1, 10)
                if subject != config.subject_id
            )
        )
        training_session = concatenate_sessions(
            [
                prepare_session(config, subject, "T")
                for subject in source_subjects
            ]
        )
        test_session = prepare_session(
            config,
            config.subject_id,
            "E",
        )

    train_indices, validation_indices = _split_session(
        training_session,
        config.validation_fraction,
        config.seed,
    )
    return (
        training_session,
        train_indices,
        validation_indices,
        test_session,
    )


def build_model(
    config: ExperimentConfig,
    session: PreparedSession,
) -> HADGCN:
    n_times = int(session.raw_eeg.shape[-1])
    eeg_node_feature_dim = int(
        session.graph_node_features.shape[-1]
    )
    return HADGCN(
        n_chans=int(session.raw_eeg.shape[1]),
        n_times=n_times,
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


def run_experiment(
    config: ExperimentConfig,
) -> Dict[str, float]:
    config.validate()
    set_random_seed(config.seed)
    device = resolve_device()

    LOGGER.info("Device: %s", device)
    LOGGER.info("Model: HAD-GCN")
    LOGGER.info("Protocol: %s", config.protocol)
    LOGGER.info(
        "Patent placeholders | branch=%s | ECG=zeros unless externally supplied",
        config.manual_branch_placeholder,
    )

    (
        training_session,
        train_indices,
        validation_indices,
        test_session,
    ) = _prepare_protocol(config)

    # Fit supervised multiclass FBCSP only on the training split, then reuse
    # the fitted filters for validation and independent test data.
    fbcsp = FBCSPNodeFeatureExtractor(
        sampling_rate=training_session.sampling_rate,
        subbands_hz=config.graph_subbands_hz,
        n_components=config.graph_features_per_band,
        filter_order=config.filter_order,
    )
    fbcsp.fit(
        training_session.raw_eeg[train_indices],
        training_session.labels[train_indices],
    )
    training_session.graph_node_features = fbcsp.transform(
        training_session.raw_eeg
    )
    test_session.graph_node_features = fbcsp.transform(
        test_session.raw_eeg
    )

    train_dataset = HADGCNDataset(
        training_session,
        train_indices,
        config.manual_branch_placeholder,
    )
    validation_dataset = HADGCNDataset(
        training_session,
        validation_indices,
        config.manual_branch_placeholder,
    )
    test_dataset = HADGCNDataset(
        test_session,
        None,
        config.manual_branch_placeholder,
    )

    train_loader = make_loader(
        train_dataset,
        config.batch_size,
        True,
        device,
        config.num_workers,
    )
    validation_loader = make_loader(
        validation_dataset,
        config.batch_size,
        False,
        device,
        config.num_workers,
    )
    test_loader = make_loader(
        test_dataset,
        config.batch_size,
        False,
        device,
        config.num_workers,
    )

    model = build_model(config, training_session)
    LOGGER.info(
        "Actual trainable parameters: %d",
        count_trainable_parameters(model),
    )
    LOGGER.info(
        "Top-level parameter counts: %s",
        module_parameter_counts(model),
    )

    history = fit(
        model=model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        device=device,
        epochs=config.epochs,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        early_stopping_enabled=config.early_stopping_enabled,
        patience=config.patience,
        log_period=config.log_period,
    )

    criterion = nn.CrossEntropyLoss()
    test_metrics = evaluate(
        model,
        test_loader,
        criterion,
        device,
    )
    LOGGER.info(
        "Test | loss %.4f | accuracy %.2f%% | kappa %.4f",
        test_metrics["loss"],
        test_metrics["accuracy"] * 100,
        test_metrics["kappa"],
    )

    output_dir = (
        config.output_folder
        / "subject_{:02d}".format(config.subject_id)
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    results: Dict[str, float] = {
        "test_loss": float(test_metrics["loss"]),
        "test_accuracy": float(test_metrics["accuracy"]),
        "test_kappa": float(test_metrics["kappa"]),
        "trainable_parameters": float(
            count_trainable_parameters(model)
        ),
    }

    calibration_state = None
    if config.calibration_enabled:
        validation_logits, validation_labels, validation_cz = (
            collect_outputs(
                model,
                validation_loader,
                device,
            )
        )
        test_logits, test_labels, test_cz = collect_outputs(
            model,
            test_loader,
            device,
        )

        calibrator = train_calibrator(
            validation_logits=validation_logits,
            validation_labels=validation_labels,
            validation_cz_signals=validation_cz,
            n_classes=config.n_classes,
            hidden_dim=config.calibration_hidden_dim,
            top_k=config.calibration_top_k,
            epochs=config.calibration_epochs,
            learning_rate=config.calibration_learning_rate,
            device=device,
        )
        calibrated_logits, temperatures = apply_calibrator(
            calibrator,
            test_logits,
            test_cz,
            config.calibration_top_k,
            device,
        )
        calibrated_metrics = calibration_metrics(
            calibrated_logits,
            test_labels,
            config.confidence_threshold,
        )
        results.update(
            {
                "calibrated_accuracy": calibrated_metrics[
                    "accuracy"
                ],
                "calibrated_kappa": calibrated_metrics["kappa"],
                "calibrated_coverage": calibrated_metrics[
                    "coverage"
                ],
                "calibrated_mean_confidence": calibrated_metrics[
                    "mean_confidence"
                ],
                "mean_temperature": float(
                    np.mean(temperatures)
                ),
            }
        )
        calibration_state = calibrator.state_dict()

    torch.save(
        {
            "model_name": "HAD-GCN",
            "model_state_dict": model.state_dict(),
            "calibration_state_dict": calibration_state,
            "config": asdict(config),
            "results": results,
            "parameter_counts": module_parameter_counts(model),
            "patent_placeholders": {
                "TF_BRANCH_SELECTION_PLACEHOLDER": (
                    config.manual_branch_placeholder
                ),
                "ECG_ADAPTIVE_FEATURES_PLACEHOLDER": None,
            },
        },
        output_dir / "had_gcn.pt",
    )
    save_history(
        history,
        output_dir / "history.csv",
    )
    save_json(
        results,
        output_dir / "metrics.json",
    )
    return results
