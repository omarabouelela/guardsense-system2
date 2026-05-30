from __future__ import annotations

from pathlib import Path
from typing import Any

from src.data.video_preprocess import VideoProcessConfig
from src.fusion.decision_logic import FusionRuntime, FusionRuntimeConfig, FusionThresholds
from src.fusion.event_schema import FusionEvent
from src.fusion.frigate_adapter import (
    FRIGATE_CLIP_METADATA_KEY,
    FRIGATE_POSE_METADATA_KEY,
    FrigateAdapter,
    FrigateAdapterConfig,
)
from src.scripts.run_dual_inference import attach_frigate_pose


class _FakeResponse:
    def __init__(self, body: bytes) -> None:
        self.body = body

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *_: Any) -> None:
        return None

    def read(self) -> bytes:
        return self.body


def _event(event_id: str = "abc/123") -> FusionEvent:
    return FusionEvent(
        event_id=event_id,
        camera_name="front",
        timestamp_start=0.0,
        source="frigate_api",
    )


def test_attach_event_clip_downloads_safe_local_clip(tmp_path: Path, monkeypatch: Any) -> None:
    adapter = FrigateAdapter(FrigateAdapterConfig(api_base_url="http://frigate.local"))
    seen_urls: list[str] = []

    def fake_urlopen(request: Any, timeout: int) -> _FakeResponse:
        seen_urls.append(request.full_url)
        assert timeout == 30
        return _FakeResponse(b"clip-bytes")

    monkeypatch.setattr("src.fusion.frigate_adapter.urlopen", fake_urlopen)

    event = adapter.attach_event_clip(_event(), tmp_path)

    expected_path = tmp_path / "frigate_abc_123_clip.mp4"
    assert seen_urls == ["http://frigate.local/api/events/abc%2F123/clip.mp4"]
    assert event.clip_path == str(expected_path)
    assert expected_path.read_bytes() == b"clip-bytes"
    assert event.metadata[FRIGATE_CLIP_METADATA_KEY]["downloaded"] is True
    assert event.metadata[FRIGATE_CLIP_METADATA_KEY]["output_path"] == str(expected_path)


def test_attach_event_clip_dry_run_only_records_plan(tmp_path: Path, monkeypatch: Any) -> None:
    adapter = FrigateAdapter(FrigateAdapterConfig(api_base_url="http://frigate.local"))

    def fail_urlopen(*_: Any, **__: Any) -> None:
        raise AssertionError("dry-run should not download")

    monkeypatch.setattr("src.fusion.frigate_adapter.urlopen", fail_urlopen)

    event = adapter.attach_event_clip(_event("event:one"), tmp_path, dry_run=True)

    planned_path = tmp_path / "frigate_event_one_clip.mp4"
    assert event.clip_path is None
    assert not planned_path.exists()
    assert event.metadata[FRIGATE_CLIP_METADATA_KEY] == {
        "clip_url": "http://frigate.local/api/events/event%3Aone/clip.mp4",
        "output_path": str(planned_path),
        "downloaded": False,
        "dry_run": True,
    }


def test_attach_frigate_pose_dry_run_records_planned_pose_path(tmp_path: Path) -> None:
    adapter = FrigateAdapter(FrigateAdapterConfig(api_base_url="http://frigate.local"))

    event = attach_frigate_pose(_event("event:one"), adapter, tmp_path, raw_cfg={}, dry_run=True)

    planned_path = tmp_path / "extracted_pose" / "event_one.txt"
    assert event.pose_input_path is None
    assert not planned_path.exists()
    assert event.metadata[FRIGATE_POSE_METADATA_KEY] == {
        "output_path": str(planned_path),
        "extracted": False,
        "dry_run": True,
    }


def test_attach_frigate_pose_extracts_from_clip(tmp_path: Path, monkeypatch: Any) -> None:
    adapter = FrigateAdapter(FrigateAdapterConfig(api_base_url="http://frigate.local"))
    clip_path = tmp_path / "clip.mp4"
    clip_path.write_bytes(b"clip")
    event = _event("event:two")
    event.clip_path = str(clip_path)

    def fake_load_yolo_model(model_path: str) -> object:
        assert model_path == "yolov8n-pose.pt"
        return object()

    def fake_extract_video_pose(**kwargs: Any) -> tuple[bool, int, int, float, str]:
        assert kwargs["video_path"] == clip_path
        kwargs["output_path"].write_text("pose\n", encoding="utf-8")
        return True, 12, 10, 1.0, "ok"

    monkeypatch.setattr("src.scripts.run_dual_inference.load_yolo_model", fake_load_yolo_model)
    monkeypatch.setattr("src.scripts.run_dual_inference.extract_video_pose", fake_extract_video_pose)

    event = attach_frigate_pose(event, adapter, tmp_path, raw_cfg={"runtime": {"device": "cpu"}}, dry_run=False)

    expected_path = tmp_path / "extracted_pose" / "event_two.txt"
    assert event.pose_input_path == str(expected_path)
    assert expected_path.read_text(encoding="utf-8") == "pose\n"
    assert event.metadata[FRIGATE_POSE_METADATA_KEY]["extracted"] is True
    assert event.metadata[FRIGATE_POSE_METADATA_KEY]["frame_count"] == 12
    assert event.metadata[FRIGATE_POSE_METADATA_KEY]["detected_frame_count"] == 10


class _Trigger:
    def infer_frigate_event(self, *_: Any, **__: Any) -> dict[str, Any]:
        return {
            "class_probabilities": {"0": 0.99, "1": 0.01},
            "predicted_label": 0,
            "confidence": 0.99,
        }


class _Verifier:
    def infer_clip(self, *_: Any, **__: Any) -> dict[str, Any]:
        return {
            "class_probabilities": {"0": 0.1, "1": 0.9},
            "predicted_label": 1,
            "confidence": 0.9,
        }


def test_frigate_downloaded_clip_still_runs_verifier_after_trigger(tmp_path: Path) -> None:
    clip_path = tmp_path / "clip.mp4"
    pose_path = tmp_path / "pose.txt"
    clip_path.write_bytes(b"clip")
    pose_path.write_text("pose\n", encoding="utf-8")

    event = _event("event-three")
    event.clip_path = str(clip_path)
    event.pose_input_path = str(pose_path)
    event.metadata[FRIGATE_CLIP_METADATA_KEY] = {"downloaded": True}
    event.metadata[FRIGATE_POSE_METADATA_KEY] = {"extracted": True, "output_path": str(pose_path)}
    runtime = FusionRuntime(
        trigger_inferencer=_Trigger(),
        verifier_inferencer=_Verifier(),
        config=FusionRuntimeConfig(
            extraction_output_dir=tmp_path,
            video_process=VideoProcessConfig(),
            thresholds=FusionThresholds(),
            drop_trigger_normal_early=True,
        ),
    )

    decision = runtime.process_event(event)

    assert decision.decision_stage == "verifier"
    assert decision.final_label == 1
    assert "Frigate clip downloaded" in decision.notes
    assert "pose extracted from Frigate clip" in decision.notes
    assert "verifier ran from Frigate clip" in decision.notes
