# Final report (YOLO + SAHI + time efficiency)

Date: 2026-01-17
Workspace: `/Users/mohammedrashid/Downloads/reasearch sahi`

## Goal

You wanted a reliable setup where you can:

- Train on your dataset
- Benchmark speed vs accuracy
- Visually check outputs on images and video
- Run real-time detection without the “tile scanning” behavior

This workspace now focuses on **YOLO + SAHI** only.

## Recommended workflow (UI)

Start the UI:

`".venv/bin/python" -m streamlit run ui/streamlit_app.py`

Then use:

- **Train + Benchmark** to train YOLO and record timings
- **Visualize (Image)** to upload an image and see boxes
- **Visualize (Video/Webcam)** to test on video/webcam
- **Results Tables** to view summaries

## CLI commands (optional)

### 1) Train YOLO and record timing

`".venv/bin/python" benchmarks/run_benchmark.py --data coco8.yaml --epochs 3 --device cpu --yolo-weights yolov8s.pt`

Outputs:

- `results/benchmark_summary.md`
- `results/<timestamp>_benchmark.json`
- `results/runs/**/weights/best.pt`

### 2) Speed up SAHI with pseudo-label ROI gating

`".venv/bin/python" benchmarks/roi_sahi.py --data coco8.yaml --model-path results/runs/<your_run>/weights/best.pt --device cpu`

### 3) Real-time smooth boxes (recommended)

`".venv/bin/python" benchmarks/realtime_roi_video.py --source 0 --show --device cpu \
  --proposer yolov8n.pt --proposer-imgsz 320 \
  --refiner yolov8s.pt --refiner-imgsz 640 \
  --vehicle-classes 2,3,5,7 --refresh-every 10 --max-rois 6 --roi-margin 64`

## Why SAHI sometimes looks “slow” on video

SAHI full-frame slicing processes tiles sequentially, so boxes may appear progressively.
For real-time, use the ROI pipeline (proposer + refiner) to get instant stable boxes.
