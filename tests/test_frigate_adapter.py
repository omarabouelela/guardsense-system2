from __future__ import annotations

from pathlib import Path
from typing import Any

from src.fusion.event_schema import FusionEvent
from src.fusion.frigate_adapter import FRIGATE_CLIP_METADATA_KEY, FrigateAdapter, FrigateAdapterConfig


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
