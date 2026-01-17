from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from _benchmark_utils import detect_device, ensure_ultralytics_data_dir


@dataclass
class Box:
    x1: int
    y1: int
    x2: int
    y2: int
    conf: float
    cls: int


def _clip(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def _expand_box(b: Box, margin: int, W: int, H: int) -> Box:
    return Box(
        x1=_clip(b.x1 - margin, 0, W),
        y1=_clip(b.y1 - margin, 0, H),
        x2=_clip(b.x2 + margin, 0, W),
        y2=_clip(b.y2 + margin, 0, H),
        conf=b.conf,
        cls=b.cls,
    )


def _xyxy_from_ultralytics(result) -> list[Box]:
    if result.boxes is None or len(result.boxes) == 0:
        return []
    xyxy = result.boxes.xyxy.cpu().numpy()
    conf = result.boxes.conf.cpu().numpy()
    cls = result.boxes.cls.cpu().numpy().astype(int)
    out: list[Box] = []
    for (x1, y1, x2, y2), c, k in zip(xyxy, conf, cls):
        out.append(Box(int(x1), int(y1), int(x2), int(y2), float(c), int(k)))
    return out


def _filter_classes(boxes: list[Box], keep: set[int] | None) -> list[Box]:
    if keep is None:
        return boxes
    return [b for b in boxes if b.cls in keep]


def _draw_boxes(frame_bgr: np.ndarray, boxes: list[Box], color=(0, 255, 0), label_prefix: str = "") -> None:
    for b in boxes:
        cv2.rectangle(frame_bgr, (b.x1, b.y1), (b.x2, b.y2), color, 2)
        txt = f"{label_prefix}{b.cls}:{b.conf:.2f}"
        cv2.putText(frame_bgr, txt, (b.x1, max(10, b.y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


def main() -> int:
    ensure_ultralytics_data_dir()

    ap = argparse.ArgumentParser(
        description=(
            "Real-time ROI detection: run a fast low-res proposal on the full frame, then run high-accuracy "
            "detection only on the proposed ROIs (vehicles), instead of scanning the whole image with slices."
        )
    )
    ap.add_argument("--source", default="0", help="Webcam index (e.g. 0) or path to video file")
    ap.add_argument("--device", default=None, help="cpu | mps | cuda:0 (default: auto)")

    # Stage 1: proposer (fast full-frame)
    ap.add_argument("--proposer", default="yolov8n.pt", help="Fast model for coarse proposals")
    ap.add_argument("--proposer-imgsz", type=int, default=320)
    ap.add_argument("--proposer-conf", type=float, default=0.25)

    # Stage 2: refiner (accurate per-ROI)
    ap.add_argument("--refiner", default="yolov8n.pt", help="Accurate model for ROI detection")
    ap.add_argument("--refiner-imgsz", type=int, default=640)
    ap.add_argument("--refiner-conf", type=float, default=0.25)

    ap.add_argument(
        "--vehicle-classes",
        default="2,3,5,7",
        help="COCO vehicle class IDs to keep: 2=car, 3=motorcycle, 5=bus, 7=truck. Use empty to keep all.",
    )

    ap.add_argument("--roi-margin", type=int, default=64, help="Pixels to expand each proposed ROI")
    ap.add_argument("--max-rois", type=int, default=6)
    ap.add_argument("--refresh-every", type=int, default=10, help="Run proposer every N frames (smaller=more robust)")
    ap.add_argument("--show", action="store_true", help="Show a window with detections")

    args = ap.parse_args()

    device = args.device or detect_device()
    if device == "mps":
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    keep_classes: set[int] | None
    if str(args.vehicle_classes).strip() == "":
        keep_classes = None
    else:
        keep_classes = {int(x.strip()) for x in str(args.vehicle_classes).split(",") if x.strip()}

    # Video source
    src: Any
    if str(args.source).isdigit():
        src = int(args.source)
    else:
        src = str(Path(args.source))

    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {args.source}")

    from ultralytics import YOLO

    proposer = YOLO(args.proposer)
    refiner = YOLO(args.refiner)

    frame_idx = 0
    rois: list[Box] = []

    # rolling stats
    proposer_ms: list[float] = []
    refiner_ms: list[float] = []
    total_ms: list[float] = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1

        H, W = frame.shape[:2]
        t0 = time.perf_counter()

        # Stage 1: occasional full-frame proposal (fast, low-res)
        stage1_ms = 0.0
        if (frame_idx == 1) or (args.refresh_every > 0 and frame_idx % args.refresh_every == 0) or not rois:
            s1 = time.perf_counter()
            r = proposer.predict(frame, imgsz=args.proposer_imgsz, conf=args.proposer_conf, device=device, verbose=False)[0]
            boxes = _filter_classes(_xyxy_from_ultralytics(r), keep_classes)
            boxes.sort(key=lambda b: b.conf, reverse=True)
            rois = boxes[: args.max_rois]
            rois = [_expand_box(b, args.roi_margin, W=W, H=H) for b in rois]
            stage1_ms = (time.perf_counter() - s1) * 1000
            proposer_ms.append(stage1_ms)

        # Stage 2: detect only inside ROIs (high-res crops)
        s2 = time.perf_counter()
        refined: list[Box] = []
        for roi in rois:
            crop = frame[roi.y1 : roi.y2, roi.x1 : roi.x2]
            if crop.size == 0:
                continue
            rr = refiner.predict(crop, imgsz=args.refiner_imgsz, conf=args.refiner_conf, device=device, verbose=False)[0]
            boxes = _filter_classes(_xyxy_from_ultralytics(rr), keep_classes)
            # shift boxes back to full frame
            for b in boxes:
                refined.append(
                    Box(
                        x1=b.x1 + roi.x1,
                        y1=b.y1 + roi.y1,
                        x2=b.x2 + roi.x1,
                        y2=b.y2 + roi.y1,
                        conf=b.conf,
                        cls=b.cls,
                    )
                )

        stage2_ms = (time.perf_counter() - s2) * 1000
        refiner_ms.append(stage2_ms)

        dt_ms = (time.perf_counter() - t0) * 1000
        total_ms.append(dt_ms)

        # Update ROIs from refined boxes (keeps ROIs following the object)
        if refined:
            refined.sort(key=lambda b: b.conf, reverse=True)
            rois = [_expand_box(b, args.roi_margin, W=W, H=H) for b in refined[: args.max_rois]]

        if args.show:
            vis = frame.copy()
            _draw_boxes(vis, rois, color=(255, 0, 0), label_prefix="ROI ")
            _draw_boxes(vis, refined, color=(0, 255, 0), label_prefix="")
            text = f"frame={frame_idx} total={dt_ms:.1f}ms (s1={stage1_ms:.1f}ms, s2={stage2_ms:.1f}ms)"
            cv2.putText(vis, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow("ROI detection", vis)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        if frame_idx % 30 == 0:
            def _mean(xs: list[float]) -> float:
                return sum(xs) / len(xs) if xs else 0.0

            print(
                f"[{frame_idx}] mean total={_mean(total_ms[-30:]):.1f}ms | "
                f"stage1={_mean(proposer_ms[-max(1, min(len(proposer_ms), 30)):]):.1f}ms | "
                f"stage2={_mean(refiner_ms[-30:]):.1f}ms | rois={len(rois)}"
            )

    cap.release()
    if args.show:
        cv2.destroyAllWindows()

    if total_ms:
        avg = sum(total_ms) / len(total_ms)
        fps = 1000.0 / avg if avg > 0 else 0.0
        print(f"Done. Avg latency: {avg:.1f}ms/frame (~{fps:.1f} FPS)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
