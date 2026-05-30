from __future__ import annotations

from src.scripts.run_rtsp_realtime_pipeline import (
    RealtimeThresholds,
    build_loop_config,
    build_parser,
    classify_realtime_decision,
    mask_rtsp_url,
    safe_filename_token,
)


def test_mask_rtsp_url_hides_credentials() -> None:
    masked = mask_rtsp_url("rtsp://user:secret@example.local:554/live/stream?channel=1")

    assert masked == "rtsp://***:***@example.local:554/live/stream?channel=1"
    assert "user" not in masked
    assert "secret" not in masked


def test_mask_rtsp_url_without_credentials_is_unchanged() -> None:
    assert mask_rtsp_url("rtsp://example.local:554/live") == "rtsp://example.local:554/live"


def test_realtime_threshold_ladder() -> None:
    thresholds = RealtimeThresholds(trigger_medium=0.55, trigger_high=0.75, verifier_alert=0.82)

    assert classify_realtime_decision(0.54, None, thresholds) == "normal_background"
    assert classify_realtime_decision(0.55, None, thresholds) == "low_confidence_suspicious_motion"
    assert classify_realtime_decision(0.74, None, thresholds) == "low_confidence_suspicious_motion"
    assert classify_realtime_decision(0.75, None, thresholds) == "trigger_positive"
    assert classify_realtime_decision(0.90, 0.81, thresholds) == "verifier_rejected"
    assert classify_realtime_decision(0.90, 0.82, thresholds) == "confirmed_suspicious"


def test_safe_filename_token_removes_url_sensitive_characters() -> None:
    assert safe_filename_token("classroom/cam:01@example") == "classroom_cam_01_example"


def test_cli_threshold_defaults() -> None:
    args = build_parser().parse_args(
        [
            "--config",
            "configs/fusion_runtime_clean_rwf_balanced.yaml",
            "--rtsp-url",
            "rtsp://example.local/live",
            "--camera-name",
            "classroom_cam",
            "--output-dir",
            "artifacts/live_runs/test",
        ]
    )

    assert args.trigger_high_threshold == 0.75
    assert args.verifier_alert_threshold == 0.82


def test_cli_threshold_values_override_yaml_thresholds() -> None:
    raw_cfg = {
        "rtsp_realtime": {
            "thresholds": {
                "trigger_medium": 0.50,
                "trigger_high": 0.99,
                "verifier_alert": 0.99,
            }
        }
    }

    loop_cfg = build_loop_config(raw_cfg, trigger_high_threshold=0.70, verifier_alert_threshold=0.80)

    assert loop_cfg.thresholds.trigger_medium == 0.50
    assert loop_cfg.thresholds.trigger_high == 0.70
    assert loop_cfg.thresholds.verifier_alert == 0.80
