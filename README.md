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
- Device config `device: auto` resolves to `cuda` when available, otherwise `cpu`.

### Supported data inputs
- YOLOv8-pose style `.txt`
- `.npy`
- `.hdf5` / `.h5`

For `.txt` Trigger inputs, both layouts are supported:
- `sequence_per_file`: one txt file contains many temporal rows.
- `frame_per_file`: one txt file per frame, grouped deterministically by folder + filename prefix.
- `auto`: detects likely layout (default).

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
# Both command styles are supported:
# 1) module execution
python -m src.scripts.prepare_trigger_data --config configs/trigger_data.yaml
python -m src.scripts.train_trigger --config configs/trigger_train.yaml
python -m src.scripts.eval_trigger --config configs/trigger_eval.yaml

# 2) direct script execution
python src/scripts/prepare_trigger_data.py --config configs/trigger_data.yaml
python src/scripts/train_trigger.py --config configs/trigger_train.yaml
python src/scripts/eval_trigger.py --config configs/trigger_eval.yaml
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
- Device config `device: auto` resolves to `cuda` when available, otherwise `cpu`.

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

### Torchvision compatibility note (Python 3.12)
- Verifier training/eval/inference requires `torchvision` video components (`torchvision.models.video` and `torchvision.io.read_video`).
- If import fails, install a matching PyTorch + torchvision pair from official wheels for your Python/CUDA build.

## Fusion Runtime (Trigger + Verifier + Frigate)

A production-style fusion runtime is available for downstream Frigate event inference.

### Files
- Runtime schema and IO helpers: `src/fusion/event_schema.py`
- Fusion decision logic and thresholds: `src/fusion/decision_logic.py`
- Frigate API/MQTT adapters: `src/fusion/frigate_adapter.py`
- CLI runner: `src/scripts/run_dual_inference.py`
- Runtime config template: `configs/fusion_runtime.yaml`
- Example event input: `configs/fusion_example_event.json`
- Example keyed output: `configs/fusion_example_output.json`

### Supported event sources
- Frigate event ID via API lookup
- Frigate MQTT-like JSON payload
- Local JSON payload file(s)
- Offline CSV manifest rows

### Runtime behavior summary
1. Ingest Frigate-compatible event payload(s).
2. Run Trigger first when pose input exists.
3. Escalate suspicious/uncertain events to Verifier according to configurable thresholds.
4. Resolve Verifier clip from event clip directly or extract centered clip from recording.
5. Fuse Trigger + Verifier into final class `0/1/2` with explainable notes.
6. Emit JSON keyed by `event_id -> camera_name` and run artifacts.

### Checkpoint path resolution
- Training writes timestamped directories under `artifacts/runs/` and `artifacts/verifier_runs/`.
- After each training run, `latest/` is refreshed so default eval/runtime configs can resolve checkpoints automatically.
- Eval/runtime also fall back to the most recent valid run checkpoint when configured `latest` paths are missing.

### Raw data path compatibility
- Canonical config convention is `dataset/raw/...`.
- Backward compatibility is preserved: `data/raw/...` is automatically resolved when possible.

### Output artifacts
Each run writes:
- `run.log`
- `processed_events.jsonl`
- `failed_events.jsonl`
- `review_needed.jsonl`
- `summary.json`
- `final_output_keyed.json`

### Example CLI commands

```bash
# 1) Dry-run with example JSON event (no model execution)
python -m src.scripts.run_dual_inference \
  --config configs/fusion_runtime.yaml \
  --event-json configs/fusion_example_event.json \
  --output-dir artifacts/fusion_dry_run \
  --dry-run --debug

# 2) Process one Frigate event ID via API
python -m src.scripts.run_dual_inference \
  --config configs/fusion_runtime.yaml \
  --event-id 1709851020.457391-abc123 \
  --output-dir artifacts/fusion_event

# 3) Batch from manifest CSV
python -m src.scripts.run_dual_inference \
  --config configs/fusion_runtime.yaml \
  --manifest-csv data/fusion_manifest.csv \
  --output-dir artifacts/fusion_manifest
```

## End-to-end orchestration

Use `src/scripts/run_full_pipeline.py` to run data prep, training, evaluation, and runtime inference with optional resume/skip/dry-run controls.

```bash
python -m src.scripts.run_full_pipeline \
  --config configs/runtime.yaml \
  --output-dir artifacts/full_pipeline \
  --dry-run
```

Outputs include:
- `final_experiment_report.json`
- `final_experiment_report.csv`
- `final_experiment_report.txt`
- stage marker files (`*.done`) for resume mode

This keeps Frigate downstream integration intact and emits runtime outputs keyed by `event_id` and `camera_name`.

## Quick smoke checks

```bash
python -m unittest discover -s tests
python src/scripts/prepare_trigger_data.py --help
python src/scripts/train_trigger.py --help
python src/scripts/eval_trigger.py --help
python src/scripts/prepare_verifier_data.py --help
python src/scripts/train_verifier.py --help
python src/scripts/eval_verifier.py --help
python src/scripts/run_dual_inference.py --help
python src/scripts/run_full_pipeline.py --help
python -m src.scripts.run_dual_inference --config configs/fusion_runtime.yaml --event-json configs/fusion_example_event.json --output-dir artifacts/fusion_dry_run --dry-run
python -m src.scripts.run_full_pipeline --config configs/runtime.yaml --output-dir artifacts/full_pipeline_dry_run --dry-run
```
