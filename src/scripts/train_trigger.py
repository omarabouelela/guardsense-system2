"""CLI for Trigger model training."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from src.trigger.model import TriggerModelConfig
from src.trigger.train import TrainConfig, train_trigger


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(description="Train GuardSense Trigger model")
    parser.add_argument("--config", type=Path, required=True, help="Training config YAML")
    return parser


def main() -> None:
    """Main CLI."""
    args = build_parser().parse_args()
    raw = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    model_cfg = TriggerModelConfig(**raw.get("model", {}))
    cfg = TrainConfig(
        dataset_path=raw["dataset_path"],
        output_dir=raw["output_dir"],
        batch_size=raw.get("batch_size", 64),
        epochs=raw.get("epochs", 40),
        lr=raw.get("lr", 1e-3),
        weight_decay=raw.get("weight_decay", 1e-4),
        patience=raw.get("patience", 8),
        seed=raw.get("seed", 42),
        num_workers=raw.get("num_workers", 0),
        device=raw.get("device", "auto"),
        class_weights=raw.get("class_weights"),
        model=model_cfg,
    )
    result = train_trigger(cfg)
    print(result)


if __name__ == "__main__":
    main()
