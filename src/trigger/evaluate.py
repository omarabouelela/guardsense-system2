"""Evaluation helpers for Trigger model checkpoints."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.data.trigger_dataset import TriggerPoseDataset
from src.common.runtime_utils import resolve_checkpoint_path, resolve_device
from src.trigger.model import TriggerModelConfig, build_trigger_model
from src.trigger.train import macro_metrics


@dataclass(slots=True)
class EvaluateConfig:
    """Evaluation config for a trained Trigger model."""

    model_path: str
    dataset_path: str
    split: str = "test"
    batch_size: int = 128
    device: str = "auto"
    output_path: str | None = None


def evaluate_trigger(config: EvaluateConfig) -> dict[str, Any]:
    """Evaluate model checkpoint on selected split from HDF5 dataset."""
    device = resolve_device(config.device)
    model_path = resolve_checkpoint_path(config.model_path, base_dir="artifacts/runs", run_prefix="trigger_")

    with h5py.File(config.dataset_path, "r") as h5f:
        x = h5f[f"X_{config.split}"][()]
        y = h5f[f"y_{config.split}"][()]

    ds = TriggerPoseDataset(x, y)
    loader = DataLoader(ds, batch_size=config.batch_size, shuffle=False)

    checkpoint = torch.load(model_path, map_location=device)
    model_cfg = TriggerModelConfig(**checkpoint.get("config", {}))
    model = build_trigger_model(model_cfg).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    criterion = nn.CrossEntropyLoss()

    model.eval()
    losses: list[float] = []
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            losses.append(float(loss.item()))
            preds = torch.argmax(logits, dim=-1)
            y_true.append(batch_y.cpu().numpy())
            y_pred.append(preds.cpu().numpy())

    y_true_np = np.concatenate(y_true)
    y_pred_np = np.concatenate(y_pred)
    metrics = macro_metrics(y_true_np, y_pred_np, num_classes=model_cfg.num_classes)
    metrics["loss"] = float(np.mean(losses)) if losses else 0.0

    if config.output_path:
        output_path = Path(config.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    return metrics
