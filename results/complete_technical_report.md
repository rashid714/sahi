# Complete Technical Report — YOLO + SAHI (Sliced Inference) + ROI Optimization + UI

Date: 2026-01-17  
Workspace: `/Users/mohammedrashid/Downloads/reasearch sahi`

## 1) What this project does (scope)

This workspace provides an end-to-end, reproducible pipeline to:

- Train **Ultralytics YOLOv8** on a dataset defined by an Ultralytics YAML
- Benchmark **speed + key accuracy metrics**
- Compare inference latency for:
  - **Ultralytics full-frame inference**
  - **SAHI sliced inference** (`get_sliced_prediction`)
- Speed up large-image inference using **ROI gating** via **pseudo-label proposals**
- Run a **real-time video pipeline** (proposer + refiner) for stable, low-latency boxes
- Use a **Streamlit UI** for training, visual inspection (image/video), dataset import, and exporting outputs

This project is **YOLO + SAHI only**.

## 2) Tech stack used

**Core ML / vision**
- Ultralytics YOLOv8 (`ultralytics`)
- SAHI (`sahi`) for sliced inference and postprocess
- PyTorch (`torch`) backend
- OpenCV (`opencv-python`) for video decode + real-time display

**UI / tooling**
- Streamlit (`streamlit`) for a clickable UI
- NumPy + Pillow for image handling

## 3) Environment (reproducibility)

Collected from the venv in this workspace:

- OS: macOS 26.2 (arm64)
- Python: 3.12.12
- Torch: 2.9.1
- Ultralytics: 8.4.5
- SAHI: 0.11.36
- Streamlit: 1.53.0
- OpenCV: 4.11.0.86
- Torch MPS available: True

Device notes:
- Training/inference can run on `cpu` or `mps` (Apple Silicon). For consistency, benchmarks are often more stable on `cpu`.
- Some PyTorch operations may fall back on MPS; scripts set `PYTORCH_ENABLE_MPS_FALLBACK=1` when using MPS.

## 4) Repository layout (what each folder/file is)

**UI**
- ui/streamlit_app.py — Streamlit app (dataset import, training+bench, visualize image/video, outputs gallery)
- ui/README.md — how to run the UI

**Benchmarks / scripts**
- benchmarks/run_benchmark.py — trains YOLO and benchmarks inference time vs SAHI slicing
- benchmarks/roi_sahi.py — ROI gating using pseudo-label proposals to reduce SAHI work
- benchmarks/realtime_roi_video.py — real-time proposer/refiner pipeline for webcam/video
- benchmarks/_benchmark_utils.py — shared helpers (timestamping, system info, ULTRALYTICS_DATA_DIR, timing helpers)

**Results**
- results/benchmark_summary.md — appended benchmark summary table
- results/roi_sahi_summary.md — ROI benchmark summary table
- results/final_report.md — short summary report
- results/runs/ — Ultralytics training artifacts (weights, logs)
- results/ui_outputs/ — UI saved outputs + gallery index + exports (created when you save outputs)

**Datasets**
- datasets/ — local datasets and imports

## 5) Dataset handling

### Ultralytics dataset YAML
Training/benchmarking uses an Ultralytics YAML like `coco8.yaml` or `coco128.yaml`.

- The Streamlit UI can also **import a dataset ZIP** and auto-generate `data.yaml`.
- Recommended YOLO dataset ZIP layout:
  - `images/train`, `images/val` (or `images/valid`)
  - `labels/train`, `labels/val` (or `labels/valid`)

### Dataset ZIP import (UI)
In the UI page **0) Dataset Import**:
- Upload a `.zip`
- It extracts to `datasets/imports/<dataset>_<timestamp>/`
- It generates `data.yaml` in that extracted dataset root

## 6) Training pipeline (how it works)

Implemented in benchmarks/run_benchmark.py:

- Uses `ultralytics.YOLO(weights).train(...)`
- Captures:
  - training wall time (seconds)
  - key metrics parsed from Ultralytics `results.csv` (e.g., `metrics/mAP50-95`)
- Saves weights to:
  - `results/runs/<run_name>/weights/best.pt`

## 7) Benchmarking methodology (time efficiency)

### 7.1 Full-image inference benchmark
- Loads validation images from the dataset YAML `val` entry
- Runs full-image inference using `model.predict(source=[...])`
- Records total time and converts to seconds-per-image

### 7.2 SAHI sliced inference benchmark
- Wraps YOLO weights using:
  - `AutoDetectionModel.from_pretrained(model_type='yolov8', model_path=..., confidence_threshold=..., device=...)`
- For each validation image, runs:
  - `get_sliced_prediction(..., slice_height, slice_width, overlap_*, postprocess_type, ...)`
- Records per-image time, then reports mean / median / p95

Why SAHI can be slower:
- Slicing adds overhead (tiling + merging) and may run the model many times per image.
- It is primarily used when objects are **small** or images are **very large**, where full-frame inference misses objects.

## 8) ROI gating / pseudo-label optimization (speedup strategy)

Implemented in benchmarks/roi_sahi.py:

Goal: avoid slicing the entire image when objects are sparse.

Approach:
1) Run a fast YOLO model (the “proposer”) at low resolution on the full image.
2) Convert predicted boxes into a small set of ROIs:
   - expand with margin
   - merge overlapping ROIs
   - keep top-K ROIs
3) Run SAHI only inside those ROIs (or standard inference if ROI is small).
4) If no ROI is proposed, skip heavy inference (max time savings).

This is the main “real-time friendly” concept for large images: **reduce the area you process**.

## 9) Real-time video pipeline (recommended for smooth results)

Implemented in benchmarks/realtime_roi_video.py:

Two-stage pipeline:
- Stage 1: **proposer** YOLO on full frame, low-res, every N frames
- Stage 2: **refiner** YOLO on ROIs only, higher-res, every frame

Why this works better than SAHI slicing for video:
- SAHI processes tiles sequentially; boxes can appear progressively across tiles.
- Proposer/refiner gives stable boxes with much less latency.

Outputs:
- Prints rolling latency stats and final average FPS/latency.

## 10) Streamlit UI (what it supports)

Implemented in ui/streamlit_app.py. Pages:

- **0) Dataset Import**: upload dataset ZIP and generate `data.yaml`
- **1) Train + Benchmark**: runs benchmarks/run_benchmark.py and shows latest JSON
- **2) Visualize (Image)**:
  - Ultralytics full-image inference
  - SAHI sliced inference
  - Save annotated outputs to `results/ui_outputs/` and export
- **3) Visualize (Video/Webcam)**:
  - Launch ROI pipeline (OpenCV window)
  - In-browser video preview + optional CSV timing log
- **3.5) Outputs Gallery**:
  - Review saved outputs
  - Export a ZIP of the gallery
- **4) Results Tables**:
  - Shows results/benchmark_summary.md and results/roi_sahi_summary.md

## 11) Benchmark results (from this workspace)

### 11.1 Training + speed (Ultralytics vs SAHI sliced)
Source: results/benchmark_summary.md

| timestamp_utc | data | epochs | device | model | train_s | mAP50-95 | infer_ms/img | sahi_ms/img |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 20260117T034022Z | coco8.yaml | 1 | cpu | yolo | 4.9 | 0.6504 | 47.61 | 139.06 |
| 20260117T040112Z | coco8.yaml | 3 | cpu | yolo | 59.4 | 0.7070 | 100.84 | 224.50 |

Interpretation:
- SAHI sliced inference is slower per image than full-frame inference on this dataset.
- Training time scales with epochs (expected).

### 11.2 ROI gating benchmark
Source: results/roi_sahi_summary.md

| timestamp_utc | data | model_type | device | mode | ms/img_mean | p50 | p95 | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 20260117T034754Z | coco8.yaml | yolov8 | cpu | sahi_full | 1224.39 | 1228.76 | 2336.82 | slice=512 overlap=0.2 post=NMS |
| 20260117T034754Z | coco8.yaml | yolov8 | cpu | sahi_roi_pseudolabel | 661.74 | 693.35 | 1605.61 | rois=3 skipped=1 proposer=yolov8n.pt |

Interpretation:
- ROI gating reduces mean latency substantially vs full SAHI slicing.
- It can skip work on frames/images where no ROI is proposed.

## 12) How to reproduce (commands)

### 12.1 Run the UI
From workspace root:

```bash
".venv/bin/python" -m streamlit run ui/streamlit_app.py
```

### 12.2 Run training + benchmark (CLI)

```bash
".venv/bin/python" benchmarks/run_benchmark.py \
  --data coco8.yaml \
  --epochs 3 \
  --imgsz 640 \
  --batch 8 \
  --device cpu \
  --yolo-weights yolov8s.pt \
  --conf 0.25 \
  --max-infer-images 32 \
  --sahi-slice 512 \
  --sahi-overlap 0.2 \
  --sahi-postprocess NMS
```

### 12.3 Run ROI gating benchmark

```bash
".venv/bin/python" benchmarks/roi_sahi.py \
  --data coco8.yaml \
  --model-type yolov8 \
  --model-path results/runs/<your_run>/weights/best.pt \
  --device cpu
```

### 12.4 Run real-time ROI video

```bash
".venv/bin/python" benchmarks/realtime_roi_video.py \
  --source 0 --show --device cpu \
  --proposer yolov8n.pt --proposer-imgsz 320 \
  --refiner yolov8s.pt --refiner-imgsz 640 \
  --vehicle-classes 2,3,5,7 \
  --refresh-every 10 --max-rois 6 --roi-margin 64
```

## 13) Known limitations / practical notes

- SAHI slicing is not designed for smooth real-time video overlay; it is best used for still images or batch inference.
- Video preview in Streamlit decodes only a limited number of frames for responsiveness.
- For best reproducibility, keep `device=cpu` during benchmarking.

## 14) Key files (entry points)

- ui/streamlit_app.py
- benchmarks/run_benchmark.py
- benchmarks/roi_sahi.py
- benchmarks/realtime_roi_video.py
- results/benchmark_summary.md
- results/roi_sahi_summary.md

