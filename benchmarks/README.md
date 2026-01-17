# SAHI + YOLO + SAHI benchmark

This folder contains scripts to:

1. Train a YOLO detector on the same data
2. Record training wall time + mAP
3. Compare inference time (Ultralytics full-image vs SAHI sliced)
4. Speed up SAHI using ROI gating via pseudo-labels
5. Run a real-time ROI video pipeline (proposer + refiner)

For a single consolidated write-up (results + commands + explanation), see `results/final_report.md`.

If you prefer a clickable UI (upload image, run training, see visual outputs), run:

```bash
".venv/bin/python" -m streamlit run ui/streamlit_app.py
```

Practical “best model” guidance:
- For CPU real-time: start with `yolov8n.pt` (fast), then upgrade to `yolov8s.pt` if you need accuracy.

This folder contains a small, reproducible benchmark that:

1. Trains a **YOLO detector** (Ultralytics) on a COCO-format dataset.
2. Records **training wall time** and the key detection metrics that Ultralytics reports.
3. Measures **inference time** on the same validation images:
   - standard (full-image) inference
   - SAHI sliced inference

## What “same data” means here

Both models are trained on the exact same dataset YAML (default: `coco128.yaml`).

You can change the dataset via `--data`.

## Run

From the workspace root:

```bash
".venv/bin/python" -m pip --version
".venv/bin/python" benchmarks/run_benchmark.py --help

# Recommended quick run (small dataset, few epochs)
".venv/bin/python" benchmarks/run_benchmark.py \
  --data coco128.yaml \
  --epochs 3 \
  --imgsz 640 \
  --batch 8 \
  --yolo-weights yolov8s.pt
```

## Outputs

- `results/<timestamp>_benchmark.json` – machine-readable results.
- `results/benchmark_summary.md` – human-readable summary table (appends rows).
- `results/runs/*` – raw Ultralytics run artifacts.

## Speeding up SAHI: ROI via pseudo-labels

If you only care about a specific **object of interest**, you can avoid slicing the entire image:

1. Run a fast detector at low resolution (e.g., `yolov8n.pt` at `imgsz=320`) to generate **pseudo labels**.
2. Convert those predicted boxes into 1–N **ROIs** (merge overlaps, add a margin).
3. Run SAHI only inside those ROIs (or even just standard inference if ROI is small).

This reduces the number of slices dramatically on large images with sparse objects.

## Real-time tip: don’t slice the whole frame

If you see boxes “appear from a corner then move across”, that’s usually because **SAHI processes tiles sequentially**.
For real-time video, you generally don’t want to run SAHI over the entire frame each time.

Instead, use a 2-stage pipeline:

1. **Fast proposer** on the full frame at low resolution (cheap): find likely vehicles.
2. **Refiner** runs only on those ROIs (high-res crops): produces the final boxes quickly.

Demo script:

```bash
".venv/bin/python" benchmarks/realtime_roi_video.py --source 0 --show \
  --proposer yolov8n.pt --proposer-imgsz 320 \
  --refiner yolov8n.pt --refiner-imgsz 640 \
  --vehicle-classes 2,3,5,7 --refresh-every 10 --roi-margin 64
```

Run the ROI benchmark:

```bash
 # Example with YOLO weights produced by training
".venv/bin/python" benchmarks/roi_sahi.py \
  --data coco8.yaml \
  --model-type yolov8 \
  --model-path results/runs/<your_yolo_run>/weights/best.pt \
  --max-images 16 \
  --slice 512 --overlap 0.2 --postprocess NMS
```

## Notes

- On Apple Silicon, the script will prefer `mps` if available; otherwise it uses `cpu`.
- For a stronger benchmark, increase `--epochs` and use a larger dataset (e.g., COCO 2017).
