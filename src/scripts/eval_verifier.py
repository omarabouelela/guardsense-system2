"""CLI for Verifier checkpoint evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from src.verifier.evaluate import VerifierEvalConfig, evaluate_verifier


def build_parser() -> argparse.ArgumentParser:
    """Build parser."""
    parser = argparse.ArgumentParser(description="Evaluate GuardSense Verifier model")
    parser.add_argument("--config", type=Path, required=True, help="YAML config path")
    return parser


def main() -> None:
    """CLI main."""
    args = build_parser().parse_args()
    raw = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    cfg = VerifierEvalConfig(
        model_path=raw["model_path"],
        manifest_path=raw["manifest_path"],
        split=raw.get("split", "test"),
        batch_size=raw.get("batch_size", 8),
        device=raw.get("device", "auto"),
        num_frames=raw.get("num_frames", 16),
        temporal_stride=raw.get("temporal_stride", 2),
        output_path=raw.get("output_path"),
    )
    print(evaluate_verifier(cfg))


if __name__ == "__main__":
    main()
