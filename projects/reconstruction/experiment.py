"""
Experiment implementation for the Reconstruction task, separated from core pipeline.
"""

# Standard library imports
import random
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Union

# Third-party imports
import numpy as np
import torch
import torchinfo
import wandb
import yaml

# Local imports
from cvnn.base_experiment import BaseExperiment
from cvnn.data import (
    get_dataset_info,
    get_dataset_split_indices,
)
from cvnn.evaluate import (
    create_dataset_split_mask,
    evaluate_reconstruction,
    reconstruct_full_image,
)
from cvnn.model_utils import build_model_from_config
from cvnn.plugins import register_plugin
from cvnn.utils import setup_logging
from cvnn.visualize import (
    create_dataset_split_visualization,
    plot_losses,
    show_reconstructions,
)

# initialize module-level logger
logger = setup_logging(__name__)


@register_plugin("reconstruction")
class ReconstructionExperiment(BaseExperiment):
    def build_model(self) -> torch.nn.Module:
        """Build autoencoder model for reconstruction task using model registry."""
        return build_model_from_config(self.cfg, task="reconstruction")

    def evaluate(self) -> Any:
        # reconstruct the full image
        # evaluate reconstruction metrics on test set (requires test_loader)
        assert (
            self.test_loader is not None
        ), "`test_loader` must be provided for evaluation."
        self.metrics = evaluate_reconstruction(
            self.test_loader,
            self.model,
            self.cfg,  # Pass config to evaluation dispatcher
            device=self.device,
        )
        return self.metrics

    def visualize(self) -> None:
        # Create dataset split visualization for patch-based datasets
        train_indices, valid_indices, test_indices = get_dataset_split_indices(self.cfg)

        # Use registry to check dataset capabilities
        dataset_info = get_dataset_info(self.cfg["data"]["dataset"]["name"])
        supports_full_image = dataset_info.get(
            "supports_full_image_reconstruction", False
        )

        if supports_full_image:
            mask = create_dataset_split_mask(
                cfg=self.cfg,
                full_loader=self.full_loader,
                train_indices=train_indices,
                valid_indices=valid_indices,
                test_indices=test_indices,
                nsamples_per_cols=self.nsamples_per_cols,
                nsamples_per_rows=self.nsamples_per_rows,
            )
            create_dataset_split_visualization(
                mask=mask,
                cfg=self.cfg,
                logdir=self.logdir,
                wandb_log=self.wandb_log is not None,
                train_indices=train_indices,
                valid_indices=valid_indices,
                test_indices=test_indices,
            )
        original_image, reconstruct_image = reconstruct_full_image(
            self.model,
            self.full_loader,
            config=self.cfg,
            device=self.device,
            nsamples_per_rows=self.nsamples_per_rows,
            nsamples_per_cols=self.nsamples_per_cols,
        )

        # ensure training and evaluation have run
        if self.cfg.get("mode") in ("full", "train", "retrain"):
            assert (
                self.history is not None
            ), "`train()` must be called before `visualize()`"
            plot_losses(
                self.history["train_loss"],
                self.history["valid_loss"],
                savepath=str(self.logdir / "loss_curve.png"),
                title="Training & Validation Loss",
            )
        polsar_metrics = show_reconstructions(
            original_image,
            reconstruct_image,
            logdir=self.logdir,
            wandb_log=self.wandb_log is not None,
            cfg=self.cfg,
        )
        self.metrics.update(polsar_metrics)
        return self.metrics
