from __future__ import annotations

from src.scripts.run_rtsp_realtime_pipeline import (
    RealtimeThresholds,
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
    thresholds = RealtimeThresholds(trigger_medium=0.55, trigger_high=0.75, verifier_alert=0.85)

    assert classify_realtime_decision(0.54, None, thresholds) == "normal_background"
    assert classify_realtime_decision(0.55, None, thresholds) == "low_confidence_suspicious_motion"
    assert classify_realtime_decision(0.74, None, thresholds) == "low_confidence_suspicious_motion"
    assert classify_realtime_decision(0.75, None, thresholds) == "trigger_positive"
    assert classify_realtime_decision(0.90, 0.84, thresholds) == "verifier_rejected"
    assert classify_realtime_decision(0.90, 0.85, thresholds) == "confirmed_suspicious"


def test_safe_filename_token_removes_url_sensitive_characters() -> None:
    assert safe_filename_token("classroom/cam:01@example") == "classroom_cam_01_example"
