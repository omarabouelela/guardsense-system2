# guardsense-system2

GuardSense Fit System 2 is a dual-model classroom violence-prevention stack:
- **Trigger** = pose-based early warning
- **Verifier** = RGB clip confirmation
- **Fusion** = Frigate downstream runtime
- **Labels** = 0 normal, 1 pre-fight/tension, 2 fight

## Trigger Pipeline (Pose-Based)

This repository includes a full production Trigger baseline pipeline under `src/data`, `src/trigger`, and `src/scripts`.

### Core assumptions
- Trigger model trains on pose tensors of shape **(B, T, K, C)**.
- Default is `T=32`, `K=17` (COCO), `C=3` (`x`, `y`, `visibility`).
- Temporal windows use `window=32` with `50%` overlap.
- Visibility is preserved and used in preprocessing/inference.
- Frigate stays upstream (event/media source); Trigger runs downstream.

### Supported data inputs
- YOLOv8-pose style `.txt`
- `.npy`
- `.hdf5` / `.h5`

### Implemented components
- Dataset indexing and metadata generation
- Pose parsing with malformed row handling
- Temporal window assembly
- Frame-relative and bbox-relative normalization
- Export to per-array `.npy` and master split `.hdf5`
- Split manifest, rejection logs, dataset reporting
- Pose debug plotting utilities
- PyTorch temporal-CNN baseline
- Training with class weights, scheduler, early stopping, checkpointing
- Evaluation with macro/per-class metrics and confusion matrix
- Frigate-friendly inference wrappers and internal event schema

## Configs
- `configs/trigger_data.yaml`
- `configs/trigger_train.yaml`
- `configs/trigger_eval.yaml`

## Example commands

```bash
python -m src.scripts.prepare_trigger_data --config configs/trigger_data.yaml
python -m src.scripts.train_trigger --config configs/trigger_train.yaml
python -m src.scripts.eval_trigger --config configs/trigger_eval.yaml
```

## Notes on Python compatibility
- Target runtime is Python **3.11+ / 3.12+**.
- Current development assumptions align with Python **3.12.3**.
- If PyTorch wheels for your platform lag on 3.12, pin to a stable build from the official PyTorch index.


## Verifier Pipeline (RGB Video)

This repository now includes a production Verifier baseline under `src/data`, `src/verifier`, and `src/scripts`.

### Verifier assumptions
- Trains on short RGB clips (`.mp4` / `.avi`) with labels `0/1/2`.
- Standard preprocessing target is **30 FPS**, **640x360**, and **2-4 seconds** centered on the action.
- Supports direct Frigate event clips or extraction from Frigate recordings when clips are unavailable.
- Preserves Frigate context (`event_id`, `camera_name`, optional `track_id`) in manifests and inference outputs.
- Simuletic synthetic RGB clips are treated as a Label 1 (`pre-fight/tension`) gap-filler while real data remains the backbone for labels 0 and 2.

### Implemented Verifier components
- Recursive video indexing and ffprobe metadata extraction
- Validation for unreadable, too-short, dark, and blurry clips
- Standardized preprocessing with ffmpeg
- Deterministic stratified splitting with optional source-aware grouping
- Dataset summaries and rejection reporting
- Torchvision video baseline (`r3d_18` / `mc3_18`)
- Training with class weights, scheduler, early stopping, mixed precision, and full run artifacts
- Evaluation with macro/per-class metrics and confusion matrix
- Frigate event-aware inference wrappers and fallback behavior for snapshot-only events

### Verifier configs
- `configs/verifier_data.yaml`
- `configs/verifier_train.yaml`
- `configs/verifier_eval.yaml`

### Verifier example commands

```bash
python -m src.scripts.prepare_verifier_data --config configs/verifier_data.yaml
python -m src.scripts.train_verifier --config configs/verifier_train.yaml
python -m src.scripts.eval_verifier --config configs/verifier_eval.yaml
```
