"""CLI for Verifier model training."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from src.data.verifier_dataset import VerifierDatasetConfig
from src.verifier.model import VerifierModelConfig
from src.verifier.train import VerifierTrainConfig, train_verifier


def build_parser() -> argparse.ArgumentParser:
    """Build parser."""
    parser = argparse.ArgumentParser(description="Train GuardSense Verifier model")
    parser.add_argument("--config", type=Path, required=True, help="YAML config path")
    return parser


def main() -> None:
    """CLI main."""
    args = build_parser().parse_args()
    raw = yaml.safe_load(args.config.read_text(encoding="utf-8"))

    cfg = VerifierTrainConfig(
        manifest_path=raw["manifest_path"],
        output_dir=raw["output_dir"],
        batch_size=raw.get("batch_size", 8),
        epochs=raw.get("epochs", 20),
        lr=raw.get("lr", 1e-4),
        weight_decay=raw.get("weight_decay", 1e-4),
        patience=raw.get("patience", 5),
        seed=raw.get("seed", 42),
        num_workers=raw.get("num_workers", 0),
        device=raw.get("device", "auto"),
        mixed_precision=raw.get("mixed_precision", True),
        model=VerifierModelConfig(**raw.get("model", {})),
        data=VerifierDatasetConfig(**raw.get("data", {})),
    )
    print(train_verifier(cfg))


if __name__ == "__main__":
    main()
