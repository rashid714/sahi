from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image

from _benchmark_utils import (
    append_markdown_row,
    detect_device,
    ensure_ultralytics_data_dir,
    now_utc_compact,
    system_info,
    workspace_root,
    write_json,
)


@dataclass(frozen=True)
class Roi:
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def w(self) -> int:
        return max(0, self.x2 - self.x1)

    @property
    def h(self) -> int:
        return max(0, self.y2 - self.y1)

    @property
    def area(self) -> int:
        return self.w * self.h


def _clip(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def _roi_iou(a: Roi, b: Roi) -> float:
    ix1 = max(a.x1, b.x1)
    iy1 = max(a.y1, b.y1)
    ix2 = min(a.x2, b.x2)
    iy2 = min(a.y2, b.y2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = a.area + b.area - inter
    return float(inter) / float(union) if union > 0 else 0.0


def _merge_rois(rois: list[Roi], iou_thresh: float) -> list[Roi]:
    # Greedy merge: take a ROI, merge all overlapping into a single bounding rect.
    remaining = rois[:]
    merged: list[Roi] = []
    while remaining:
        base = remaining.pop(0)
        group = [base]
        changed = True
        while changed:
            changed = False
            keep = []
            for r in remaining:
                if any(_roi_iou(r, g) >= iou_thresh for g in group):
                    group.append(r)
                    changed = True
                else:
                    keep.append(r)
            remaining = keep
        x1 = min(r.x1 for r in group)
        y1 = min(r.y1 for r in group)
        x2 = max(r.x2 for r in group)
        y2 = max(r.y2 for r in group)
        merged.append(Roi(x1, y1, x2, y2))
    return merged


def _resolve_val_images_from_data_yaml(data_yaml: str) -> list[Path]:
    from ultralytics.utils.checks import check_yaml
    import yaml

    yaml_file = Path(check_yaml(data_yaml))
    yaml_dict = yaml.safe_load(yaml_file.read_text(encoding="utf-8"))

    val_entry = yaml_dict.get("val")
    if val_entry is None:
        raise ValueError(f"Dataset YAML has no 'val' entry: {data_yaml}")

    val_path = val_entry[0] if isinstance(val_entry, list) else val_entry
    dataset_root = yaml_dict.get("path")

    if dataset_root:
        dataset_root_path = Path(dataset_root)
        if not dataset_root_path.is_absolute():
            dataset_root_path = (yaml_file.parent / dataset_root_path).resolve()
        else:
            dataset_root_path = dataset_root_path.expanduser().resolve()

        if not dataset_root_path.exists():
            dataset_name = Path(dataset_root).name
            candidate = workspace_root() / "datasets" / dataset_name
            if candidate.exists():
                dataset_root_path = candidate
    else:
        dataset_root_path = yaml_file.parent

    p = Path(str(val_path))
    val_dir = p if p.is_absolute() else (dataset_root_path / p).resolve()

    if val_dir.is_file():
        image_paths = [Path(line.strip()) for line in val_dir.read_text(encoding="utf-8").splitlines() if line.strip()]
        return [p for p in image_paths if p.exists()]

    if not val_dir.exists():
        raise FileNotFoundError(f"Could not resolve validation path from {data_yaml}: {val_dir}")

    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    return sorted([p for p in val_dir.rglob("*") if p.suffix.lower() in exts])


def propose_rois_from_pseudolabels(
    image_paths: list[Path],
    proposer_weights: str,
    device: str,
    imgsz: int,
    conf: float,
    margin: int,
    max_rois: int,
    merge_iou: float,
    target_class_ids: set[int] | None,
) -> dict[str, list[Roi]]:
    from ultralytics import YOLO

    proposer = YOLO(proposer_weights)

    rois_by_image: dict[str, list[Roi]] = {}

    # Batch predict for speed
    results = proposer.predict(
        source=[str(p) for p in image_paths],
        imgsz=imgsz,
        conf=conf,
        device=device,
        verbose=False,
    )

    for p, r in zip(image_paths, results):
        im_h, im_w = r.orig_shape
        rois: list[Roi] = []
        if r.boxes is not None and len(r.boxes) > 0:
            boxes = r.boxes.xyxy.cpu().numpy().astype(int)
            cls_ids = r.boxes.cls.cpu().numpy().astype(int)
            for (x1, y1, x2, y2), c in zip(boxes, cls_ids):
                if target_class_ids is not None and int(c) not in target_class_ids:
                    continue
                x1 = _clip(int(x1) - margin, 0, im_w)
                y1 = _clip(int(y1) - margin, 0, im_h)
                x2 = _clip(int(x2) + margin, 0, im_w)
                y2 = _clip(int(y2) + margin, 0, im_h)
                if x2 > x1 and y2 > y1:
                    rois.append(Roi(x1, y1, x2, y2))

        if rois:
            rois = _merge_rois(rois, iou_thresh=merge_iou)
            rois.sort(key=lambda rr: rr.area, reverse=True)
            rois = rois[:max_rois]

        rois_by_image[str(p)] = rois

    return rois_by_image


def sahi_roi_inference_time(
    model_type: str,
    model_path: str,
    image_paths: list[Path],
    rois_by_image: dict[str, list[Roi]],
    device: str,
    conf: float,
    slice_size: int,
    overlap: float,
    postprocess_type: str,
    perform_standard_pred: bool,
) -> dict[str, Any]:
    from sahi import AutoDetectionModel
    from sahi.predict import get_prediction, get_sliced_prediction

    detection_model = AutoDetectionModel.from_pretrained(
        model_type=model_type,
        model_path=model_path,
        confidence_threshold=conf,
        device=device,
    )

    per_image_seconds: list[float] = []
    total_rois = 0
    skipped_no_rois = 0

    for img_path in image_paths:
        pil = Image.open(img_path)
        img_np = np.asarray(pil)
        H, W = img_np.shape[0], img_np.shape[1]
        pil.close()

        rois = rois_by_image.get(str(img_path), [])
        if not rois:
            skipped_no_rois += 1
            # If no ROI found, we do nothing (fast) — this is the main efficiency gain.
            # If you prefer, you can fall back to one standard prediction here.
            per_image_seconds.append(0.0)
            continue

        t0 = time.perf_counter()
        total_rois += len(rois)

        for roi in rois:
            crop = img_np[roi.y1 : roi.y2, roi.x1 : roi.x2]
            # If crop is small, standard inference is faster than slicing.
            if crop.shape[0] <= slice_size and crop.shape[1] <= slice_size:
                _ = get_prediction(
                    crop,
                    detection_model,
                    shift_amount=[roi.x1, roi.y1],
                    full_shape=[H, W],
                    postprocess=None,
                    verbose=0,
                )
            else:
                # Sliced inference on crop; we only care about time here.
                _ = get_sliced_prediction(
                    crop,
                    detection_model,
                    slice_height=slice_size,
                    slice_width=slice_size,
                    overlap_height_ratio=overlap,
                    overlap_width_ratio=overlap,
                    perform_standard_pred=perform_standard_pred,
                    postprocess_type=postprocess_type,
                    verbose=0,
                    auto_slice_resolution=False,
                    progress_bar=False,
                )

        per_image_seconds.append(time.perf_counter() - t0)

    valid_times = [t for t in per_image_seconds if t > 0]
    if not valid_times:
        valid_times = [0.0]

    return {
        "images": len(image_paths),
        "images_skipped_no_rois": skipped_no_rois,
        "total_rois": total_rois,
        "mean_seconds_per_image": statistics.mean(per_image_seconds),
        "p50_seconds_per_image": statistics.median(per_image_seconds),
        "p95_seconds_per_image": statistics.quantiles(per_image_seconds, n=20)[18]
        if len(valid_times) >= 2
        else valid_times[0],
    }


def sahi_full_inference_time(
    model_type: str,
    model_path: str,
    image_paths: list[Path],
    device: str,
    conf: float,
    slice_size: int,
    overlap: float,
    postprocess_type: str,
    perform_standard_pred: bool,
) -> dict[str, Any]:
    from sahi import AutoDetectionModel
    from sahi.predict import get_sliced_prediction

    detection_model = AutoDetectionModel.from_pretrained(
        model_type=model_type,
        model_path=model_path,
        confidence_threshold=conf,
        device=device,
    )

    per_image_seconds: list[float] = []
    for img_path in image_paths:
        t0 = time.perf_counter()
        _ = get_sliced_prediction(
            str(img_path),
            detection_model,
            slice_height=slice_size,
            slice_width=slice_size,
            overlap_height_ratio=overlap,
            overlap_width_ratio=overlap,
            perform_standard_pred=perform_standard_pred,
            postprocess_type=postprocess_type,
            verbose=0,
            auto_slice_resolution=False,
            progress_bar=False,
        )
        per_image_seconds.append(time.perf_counter() - t0)

    return {
        "images": len(image_paths),
        "mean_seconds_per_image": statistics.mean(per_image_seconds),
        "p50_seconds_per_image": statistics.median(per_image_seconds),
        "p95_seconds_per_image": statistics.quantiles(per_image_seconds, n=20)[18]
        if len(per_image_seconds) >= 2
        else per_image_seconds[0],
    }


def main() -> int:
    ensure_ultralytics_data_dir()

    ap = argparse.ArgumentParser(description="Speed up SAHI by focusing inference on ROIs proposed by pseudo-labels")
    ap.add_argument("--data", default="coco8.yaml")
    ap.add_argument("--max-images", type=int, default=16)

    ap.add_argument("--model-type", default="yolov8", choices=["yolov8"], help="SAHI model_type")
    ap.add_argument("--model-path", required=True, help="Trained weights (e.g., best.pt)")
    ap.add_argument("--device", default=None, help="cpu | mps | cuda:0 (default: auto)")
    ap.add_argument("--conf", type=float, default=0.25)

    ap.add_argument("--slice", type=int, default=512)
    ap.add_argument("--overlap", type=float, default=0.2)
    ap.add_argument("--postprocess", default="NMS", choices=["NMS", "NMM", "GREEDYNMM"])
    ap.add_argument("--standard-pred", action="store_true", help="Do extra full-image pred alongside slices")

    ap.add_argument("--proposer-weights", default="yolov8n.pt", help="Fast model for pseudo-label ROI proposals")
    ap.add_argument("--proposer-imgsz", type=int, default=320)
    ap.add_argument("--proposer-conf", type=float, default=0.25)
    ap.add_argument("--roi-margin", type=int, default=64)
    ap.add_argument("--max-rois", type=int, default=5)
    ap.add_argument("--roi-merge-iou", type=float, default=0.2)
    ap.add_argument(
        "--target-class-ids",
        default=None,
        help="Comma-separated class ids to keep from pseudo labels (e.g., '0,2'). Default: keep all",
    )

    args = ap.parse_args()

    device = args.device or detect_device()

    # MPS note: training may hit missing backward ops; inference is generally fine.
    if device == "mps":
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    image_paths = _resolve_val_images_from_data_yaml(args.data)[: args.max_images]
    if not image_paths:
        raise ValueError("No validation images found")

    target_class_ids = None
    if args.target_class_ids:
        target_class_ids = {int(x.strip()) for x in str(args.target_class_ids).split(",") if x.strip()}

    stamp = now_utc_compact()

    rois_by_image = propose_rois_from_pseudolabels(
        image_paths=image_paths,
        proposer_weights=args.proposer_weights,
        device=device,
        imgsz=args.proposer_imgsz,
        conf=args.proposer_conf,
        margin=args.roi_margin,
        max_rois=args.max_rois,
        merge_iou=args.roi_merge_iou,
        target_class_ids=target_class_ids,
    )

    full = sahi_full_inference_time(
        model_type=args.model_type,
        model_path=args.model_path,
        image_paths=image_paths,
        device=device,
        conf=args.conf,
        slice_size=args.slice,
        overlap=args.overlap,
        postprocess_type=args.postprocess,
        perform_standard_pred=args.standard_pred,
    )

    roi = sahi_roi_inference_time(
        model_type=args.model_type,
        model_path=args.model_path,
        image_paths=image_paths,
        rois_by_image=rois_by_image,
        device=device,
        conf=args.conf,
        slice_size=args.slice,
        overlap=args.overlap,
        postprocess_type=args.postprocess,
        perform_standard_pred=args.standard_pred,
    )

    payload: dict[str, Any] = {
        "timestamp_utc": stamp,
        "system": system_info(),
        "config": {
            "data": args.data,
            "images": len(image_paths),
            "model_type": args.model_type,
            "model_path": args.model_path,
            "device": device,
            "conf": args.conf,
            "slice": args.slice,
            "overlap": args.overlap,
            "postprocess": args.postprocess,
            "standard_pred": bool(args.standard_pred),
            "proposer_weights": args.proposer_weights,
            "proposer_imgsz": args.proposer_imgsz,
            "proposer_conf": args.proposer_conf,
            "roi_margin": args.roi_margin,
            "max_rois": args.max_rois,
            "roi_merge_iou": args.roi_merge_iou,
            "target_class_ids": sorted(list(target_class_ids)) if target_class_ids else None,
        },
        "sahi_full": full,
        "sahi_roi": roi,
    }

    out_json = workspace_root() / "results" / f"{stamp}_roi_sahi.json"
    write_json(out_json, payload)

    out_md = workspace_root() / "results" / "roi_sahi_summary.md"
    header = ["timestamp_utc", "data", "model_type", "device", "mode", "ms/img_mean", "p50", "p95", "notes"]

    append_markdown_row(
        out_md,
        header,
        [
            stamp,
            str(args.data),
            args.model_type,
            device,
            "sahi_full",
            f"{full['mean_seconds_per_image'] * 1000:.2f}",
            f"{full['p50_seconds_per_image'] * 1000:.2f}",
            f"{full['p95_seconds_per_image'] * 1000:.2f}",
            f"slice={args.slice} overlap={args.overlap} post={args.postprocess}",
        ],
    )

    append_markdown_row(
        out_md,
        header,
        [
            stamp,
            str(args.data),
            args.model_type,
            device,
            "sahi_roi_pseudolabel",
            f"{roi['mean_seconds_per_image'] * 1000:.2f}",
            f"{roi['p50_seconds_per_image'] * 1000:.2f}",
            f"{roi['p95_seconds_per_image'] * 1000:.2f}",
            f"rois={roi['total_rois']} skipped={roi['images_skipped_no_rois']} proposer={Path(args.proposer_weights).name}",
        ],
    )

    print(f"Wrote: {out_json}")
    print(f"Updated: {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
