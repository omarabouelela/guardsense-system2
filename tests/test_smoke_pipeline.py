from __future__ import annotations

from pathlib import Path

from src.scripts.run_dual_inference import load_runtime_config


def test_module_imports_smoke() -> None:
    import src.trigger.evaluate  # noqa: F401
    import src.trigger.infer  # noqa: F401
    import src.trigger.train  # noqa: F401
    import src.verifier.evaluate  # noqa: F401
    import src.verifier.infer  # noqa: F401
    import src.verifier.train  # noqa: F401
    import src.fusion.decision_logic  # noqa: F401


def test_runtime_config_and_checkpoint_path_conventions() -> None:
    runtime_cfg, adapter_cfg, raw = load_runtime_config(Path("configs/fusion_runtime.yaml"), Path("artifacts/test_runtime"))
    assert runtime_cfg.video_process.target_fps == 30
    assert runtime_cfg.video_process.target_width in (640, 854)
    assert adapter_cfg.clips_root
    assert raw["runtime"]["device"] == "auto"
    assert "latest" in raw["models"]["trigger_model_path"]
    assert "latest" in raw["models"]["verifier_model_path"]
