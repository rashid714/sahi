from __future__ import annotations

import argparse
import csv
import os
import statistics
from pathlib import Path
from typing import Any

from _benchmark_utils import (
    TimedResult,
    append_markdown_row,
    detect_device,
    ensure_ultralytics_data_dir,
    now_utc_compact,
    safe_float,
    system_info,
    timed,
    workspace_root,
    write_json,
)


def _find_latest_results_csv(run_dir: Path) -> Path | None:
    candidates = sorted(run_dir.rglob("results.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def _read_last_csv_row(path: Path) -> dict[str, Any] | None:
    with path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rows[-1] if rows else None


def _pick_ckpt_dir(project_dir: Path, name: str) -> Path:
    # Ultralytics uses project/name, but may suffix like name2, name3...
    if (project_dir / name).exists():
        return project_dir / name
    # fallback: pick newest directory under project
    dirs = [p for p in project_dir.glob("*") if p.is_dir()]
    if not dirs:
        return project_dir / name
    return max(dirs, key=lambda p: p.stat().st_mtime)


def train_ultralytics(
    weights: str,
    data: str,
    epochs: int,
    imgsz: int,
    batch: int,
    device: str,
    project_dir: Path,
    name: str,
) -> dict[str, Any]:
    from ultralytics import YOLO

    model = YOLO(weights)

    train_kwargs = dict(
        data=data,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=str(project_dir),
        name=name,
        exist_ok=False,
        verbose=False,
        plots=False,
        save=True,
        val=True,
    )

    timed_train: TimedResult = timed(model.train, **train_kwargs)

    run_path = _pick_ckpt_dir(project_dir, name)
    results_csv = _find_latest_results_csv(run_path)
    last_row = _read_last_csv_row(results_csv) if results_csv else None

    weights_dir = run_path / "weights"
    best_pt = weights_dir / "best.pt"
    last_pt = weights_dir / "last.pt"

    metrics = {}
    if last_row:
        # Common keys across Ultralytics versions. If a key is missing, keep None.
        metrics = {
            "metrics/mAP50-95": safe_float(last_row.get("metrics/mAP50-95(B)"))
            or safe_float(last_row.get("metrics/mAP50-95")),
            "metrics/mAP50": safe_float(last_row.get("metrics/mAP50(B)")) or safe_float(last_row.get("metrics/mAP50")),
            "metrics/precision": safe_float(last_row.get("metrics/precision(B)"))
            or safe_float(last_row.get("metrics/precision")),
            "metrics/recall": safe_float(last_row.get("metrics/recall(B)")) or safe_float(last_row.get("metrics/recall")),
        }

    return {
        "weights": weights,
        "data": data,
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "device": device,
        "train_seconds": timed_train.seconds,
        "run_dir": str(run_path),
        "results_csv": str(results_csv) if results_csv else None,
        "best_pt": str(best_pt) if best_pt.exists() else None,
        "last_pt": str(last_pt) if last_pt.exists() else None,
        "metrics": metrics,
    }


def _resolve_val_images_from_data_yaml(data_yaml: str) -> list[Path]:
    # Use Ultralytics dataset resolution helpers when possible.
    from ultralytics.utils.checks import check_yaml
    import yaml

    yaml_file = Path(check_yaml(data_yaml))
    yaml_dict = yaml.safe_load(yaml_file.read_text(encoding="utf-8"))
    val_entry = yaml_dict.get("val")
    if val_entry is None:
        raise ValueError(f"Dataset YAML has no 'val' entry: {data_yaml}")

    # Ultralytics supports string path or list.
    val_path = val_entry[0] if isinstance(val_entry, list) else val_entry
    val_path = str(val_path)

    # Resolve relative paths using the dataset root.
    dataset_root = yaml_dict.get("path")
    if dataset_root:
        dataset_root_path = Path(dataset_root)
        if not dataset_root_path.is_absolute():
            dataset_root_path = (yaml_file.parent / dataset_root_path).resolve()
        else:
            dataset_root_path = dataset_root_path.expanduser().resolve()

        # Ultralytics built-in dataset yamls often use paths like '../datasets/coco8'.
        # In practice, Ultralytics downloads datasets under '<cwd>/datasets/<name>' by default.
        if not dataset_root_path.exists():
            dataset_name = Path(dataset_root).name
            candidate = workspace_root() / "datasets" / dataset_name
            if candidate.exists():
                dataset_root_path = candidate
    else:
        dataset_root_path = yaml_file.parent

    p = Path(val_path)
    if p.is_absolute():
        val_dir = p
    else:
        val_dir = (dataset_root_path / p).resolve()

    if val_dir.is_file():
        # Some datasets set val to a .txt list
        image_paths = [Path(line.strip()) for line in val_dir.read_text(encoding="utf-8").splitlines() if line.strip()]
        return [p for p in image_paths if p.exists()]

    if not val_dir.exists():
        raise FileNotFoundError(f"Could not resolve validation path from {data_yaml}: {val_dir}")

    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    return sorted([p for p in val_dir.rglob("*") if p.suffix.lower() in exts])


def benchmark_inference(
    weights_path: str,
    data_yaml: str,
    imgsz: int,
    conf: float,
    device: str,
    max_images: int,
    sahi_slice: int,
    sahi_overlap: float,
    sahi_postprocess: str,
    sahi_standard_pred: bool,
) -> dict[str, Any]:
    from ultralytics import YOLO

    image_paths = _resolve_val_images_from_data_yaml(data_yaml)
    image_paths = image_paths[:max_images]
    if not image_paths:
        raise ValueError("No validation images found for inference benchmark")

    # Full-image inference (Ultralytics)
    model = YOLO(weights_path)

    def _ultra_predict():
        # Passing a list triggers batch prediction.
        _ = model.predict(
            source=[str(p) for p in image_paths],
            imgsz=imgsz,
            conf=conf,
            device=device,
            verbose=False,
        )

    ultra_time = timed(_ultra_predict)

    # SAHI sliced inference
    from sahi import AutoDetectionModel
    from sahi.predict import get_sliced_prediction

    detection_model = AutoDetectionModel.from_pretrained(
        model_type="yolov8",
        model_path=weights_path,
        confidence_threshold=conf,
        device=device,
    )

    per_image_seconds = []

    for img_path in image_paths:
        t0 = timed(
            get_sliced_prediction,
            str(img_path),
            detection_model,
            slice_height=sahi_slice,
            slice_width=sahi_slice,
            overlap_height_ratio=sahi_overlap,
            overlap_width_ratio=sahi_overlap,
            postprocess_type=sahi_postprocess,
            perform_standard_pred=sahi_standard_pred,
            verbose=False,
        )
        per_image_seconds.append(t0.seconds)

    return {
        "images": len(image_paths),
        "ultralytics_total_seconds": ultra_time.seconds,
        "ultralytics_seconds_per_image": ultra_time.seconds / len(image_paths),
        "sahi_slice": sahi_slice,
        "sahi_overlap": sahi_overlap,
        "sahi_total_seconds": float(sum(per_image_seconds)),
        "sahi_seconds_per_image_mean": statistics.mean(per_image_seconds),
        "sahi_seconds_per_image_p50": statistics.median(per_image_seconds),
        "sahi_seconds_per_image_p95": statistics.quantiles(per_image_seconds, n=20)[18] if len(per_image_seconds) >= 2 else per_image_seconds[0],
    }


def main() -> int:
    ensure_ultralytics_data_dir()

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="coco128.yaml", help="Ultralytics dataset YAML (e.g., coco8.yaml, coco128.yaml)")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default=None, help="cpu | mps | cuda:0 (default: auto)")

    ap.add_argument("--yolo-weights", default="yolov8s.pt", help="YOLO weights or model name")

    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--max-infer-images", type=int, default=32)
    ap.add_argument("--sahi-slice", type=int, default=512)
    ap.add_argument("--sahi-overlap", type=float, default=0.2)
    ap.add_argument("--sahi-postprocess", default="GREEDYNMM", choices=["NMS", "NMM", "GREEDYNMM"])
    ap.add_argument("--sahi-standard-pred", action="store_true", help="Do extra full-image pred alongside slices")

    args = ap.parse_args()

    device = args.device or detect_device()

    # PyTorch MPS is still missing some backward ops for certain models (e.g., RT-DETR).
    # Enabling fallback keeps the benchmark runnable on macOS.
    if device == "mps":
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    root = workspace_root()
    runs_dir = root / "results" / "runs"
    stamp = now_utc_compact()

    results: dict[str, Any] = {
        "timestamp_utc": stamp,
        "system": system_info(),
        "ultralytics_data_dir": os.environ.get("ULTRALYTICS_DATA_DIR"),
        "config": {
            "data": args.data,
            "epochs": args.epochs,
            "imgsz": args.imgsz,
            "batch": args.batch,
            "device": device,
            "conf": args.conf,
            "max_infer_images": args.max_infer_images,
            "sahi_slice": args.sahi_slice,
            "sahi_overlap": args.sahi_overlap,
            "sahi_postprocess": args.sahi_postprocess,
            "sahi_standard_pred": bool(args.sahi_standard_pred),
        },
        "train": {},
        "inference": {},
    }

    # Train YOLO
    yolo_tag = Path(str(args.yolo_weights)).stem or "yolo"
    yolo_name = f"{yolo_tag}_{Path(args.data).stem}_e{args.epochs}_{stamp}"
    results["train"]["yolo"] = train_ultralytics(
        weights=args.yolo_weights,
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        project_dir=runs_dir,
        name=yolo_name,
    )

    yolo_ckpt = results["train"]["yolo"].get("best_pt") or results["train"]["yolo"].get("last_pt")
    if yolo_ckpt:
        results["inference"]["yolo"] = benchmark_inference(
            weights_path=yolo_ckpt,
            data_yaml=args.data,
            imgsz=args.imgsz,
            conf=args.conf,
            device=device,
            max_images=args.max_infer_images,
            sahi_slice=args.sahi_slice,
            sahi_overlap=args.sahi_overlap,
            sahi_postprocess=args.sahi_postprocess,
            sahi_standard_pred=bool(args.sahi_standard_pred),
        )

    # Persist results
    json_path = root / "results" / f"{stamp}_benchmark.json"
    write_json(json_path, results)

    # Append a compact table row
    md_path = root / "results" / "benchmark_summary.md"
    header = [
        "timestamp_utc",
        "data",
        "epochs",
        "device",
        "model",
        "train_s",
        "mAP50-95",
        "infer_ms/img",
        "sahi_ms/img",
    ]

    def add_row(model_key: str):
        tr = results["train"].get(model_key) or {}
        inf = results["inference"].get(model_key) or {}
        m = (tr.get("metrics") or {}).get("metrics/mAP50-95")
        append_markdown_row(
            md_path,
            header,
            [
                stamp,
                str(args.data),
                str(args.epochs),
                str(device),
                model_key,
                f"{tr.get('train_seconds', 0):.1f}",
                f"{m:.4f}" if isinstance(m, (int, float)) else "",
                f"{(inf.get('ultralytics_seconds_per_image', 0) * 1000):.2f}" if inf else "",
                f"{(inf.get('sahi_seconds_per_image_mean', 0) * 1000):.2f}" if inf else "",
            ],
        )

    add_row("yolo")

    print(f"Wrote: {json_path}")
    print(f"Updated: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
