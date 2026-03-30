"""CLI for Trigger model evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

try:
    from ._bootstrap import ensure_project_root_on_path
except ImportError:  # direct script execution
    from _bootstrap import ensure_project_root_on_path

ensure_project_root_on_path()

from src.trigger.evaluate import EvaluateConfig, evaluate_trigger


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(description="Evaluate GuardSense Trigger model")
    parser.add_argument("--config", type=Path, required=True, help="Evaluation config YAML")
    return parser


def main() -> None:
    """Main CLI."""
    args = build_parser().parse_args()
    raw = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    cfg = EvaluateConfig(
        model_path=raw["model_path"],
        dataset_path=raw["dataset_path"],
        split=raw.get("split", "test"),
        batch_size=raw.get("batch_size", 128),
        device=raw.get("device", "auto"),
        output_path=raw.get("output_path"),
    )
    metrics = evaluate_trigger(cfg)
    print(metrics)


if __name__ == "__main__":
    main()
