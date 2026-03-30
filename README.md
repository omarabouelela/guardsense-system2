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
