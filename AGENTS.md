# AGENTS.md

## Project overview
This repository contains GuardSense Fit System 2, a dual-model classroom violence-prevention system.

## Core rules
- Trigger model is pose-based and trains on .npy or .hdf5.
- Verifier model is video-based and trains on .mp4 or .avi.
- Labels are:
  - 0 = normal
  - 1 = pre-fight / tension
  - 2 = fight
- Frigate is the upstream event/media source.
- Do not replace Frigate logic. Work downstream of Frigate.
- Keep the repository modular and production-oriented.

## Engineering rules
- Python 3.12.3
- PyTorch
- pathlib
- type hints
- docstrings
- robust logging
- no pseudocode
- prefer maintainable code over fancy code

## Data rules
- Trigger shape: (B, T, K, C)
- T=32, K=17 COCO keypoints, C=(x,y,visibility)
- Trigger uses 32-frame windows with 50% overlap
- Verifier clips should be standardized to 30 FPS, 640x360 or 480p, 2 to 4 seconds
- Preserve visibility flags
- Normalize keypoints

## Runtime rules
- Must accept Frigate event IDs, clip paths, recording paths, and MQTT-like payloads
- Must output JSON keyed by event_id and camera_name
