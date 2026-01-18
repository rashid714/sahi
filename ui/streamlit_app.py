from __future__ import annotations

import io
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import streamlit as st


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = WORKSPACE_ROOT / "results"
BENCHMARKS_DIR = WORKSPACE_ROOT / "benchmarks"
UI_OUTPUTS_DIR = RESULTS_DIR / "ui_outputs"
UI_INDEX_PATH = UI_OUTPUTS_DIR / "index.json"


@dataclass
class RunResult:
    returncode: int
    stdout: str
    stderr: str
    seconds: float


def _python() -> str:
    return sys.executable


def _run_cmd(args: list[str], cwd: Path | None = None) -> RunResult:
    t0 = time.perf_counter()
    p = subprocess.run(
        args,
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return RunResult(
        returncode=p.returncode,
        stdout=p.stdout,
        stderr=p.stderr,
        seconds=time.perf_counter() - t0,
    )


def _find_dataset_yamls() -> list[str]:
    candidates: set[str] = set()

    # Common Ultralytics names the user has used
    for name in ["coco8.yaml", "coco128.yaml"]:
        candidates.add(name)

    # Any *.yaml under workspace/data or datasets
    for folder in [WORKSPACE_ROOT / "data", WORKSPACE_ROOT / "datasets"]:
        if folder.exists():
            for p in folder.rglob("*.yaml"):
                candidates.add(str(p.relative_to(WORKSPACE_ROOT)))

    # Also allow selecting yaml from current directory if present
    for p in WORKSPACE_ROOT.glob("*.yaml"):
        candidates.add(str(p.relative_to(WORKSPACE_ROOT)))

    return sorted(candidates)


def _find_weight_files() -> list[str]:
    """Return workspace-relative paths to common weight files."""
    out: set[str] = set()
    for name in ["yolov8n.pt", "yolov8s.pt"]:
        out.add(name)

    # Project-provided weights (best place for custom weights when deploying).
    models_dir = WORKSPACE_ROOT / "models"
    if models_dir.exists():
        for p in models_dir.rglob("*.pt"):
            out.add(str(p.relative_to(WORKSPACE_ROOT)))

    runs = WORKSPACE_ROOT / "results" / "runs"
    if runs.exists():
        for p in runs.rglob("weights/*.pt"):
            out.add(str(p.relative_to(WORKSPACE_ROOT)))
    return sorted(out)


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""


def _latest_json(pattern: str) -> Path | None:
    files = sorted(RESULTS_DIR.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def _ensure_ultralytics_data_dir():
    # Keep datasets under workspace/datasets for predictable paths.
    os.environ.setdefault("ULTRALYTICS_DATA_DIR", str((WORKSPACE_ROOT / "datasets").resolve()))
    # Keep Ultralytics cache (downloaded weights, etc.) inside the workspace when possible.
    # Some deployment environments are read-only for system locations.
    os.environ.setdefault("ULTRALYTICS_HOME", str((WORKSPACE_ROOT / ".ultralytics").resolve()))


def _resolve_model_path_for_sahi(model_path: str) -> str:
    """Resolve a model path for SAHI.

    Ultralytics can auto-download assets like 'yolov8s.pt', but SAHI expects a real file path.
    This helper tries workspace-relative paths first, then asks Ultralytics to download the asset
    and returns the resulting local file path.
    """

    p = Path(model_path)
    if p.is_file():
        return str(p)

    wp = (WORKSPACE_ROOT / model_path)
    if wp.is_file():
        return str(wp)

    try:
        from ultralytics.utils.downloads import attempt_download_asset

        downloaded = attempt_download_asset(model_path)
        return str(downloaded)
    except Exception:
        return model_path


@st.cache_resource(show_spinner=False)
def _load_yolo(model_path: str):
    from ultralytics import YOLO

    return YOLO(model_path)


@st.cache_resource(show_spinner=False)
def _load_sahi_detection_model(model_type: str, model_path: str, confidence_threshold: float, device: str):
    from sahi import AutoDetectionModel

    return AutoDetectionModel.from_pretrained(
        model_type=model_type,
        model_path=model_path,
        confidence_threshold=confidence_threshold,
        device=device,
    )


def _iou_xyxy(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    iw = max(0, inter_x2 - inter_x1)
    ih = max(0, inter_y2 - inter_y1)
    inter = iw * ih
    if inter == 0:
        return 0.0

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    denom = (area_a + area_b - inter)
    return float(inter / denom) if denom > 0 else 0.0


def _merge_rois(rois: list[tuple[int, int, int, int]], iou_threshold: float) -> list[tuple[int, int, int, int]]:
    """Greedy ROI merge by IoU (union boxes when overlapping)."""
    if not rois:
        return []

    merged: list[tuple[int, int, int, int]] = []
    for r in rois:
        rx1, ry1, rx2, ry2 = r
        did_merge = False
        for i, m in enumerate(merged):
            if _iou_xyxy(r, m) >= iou_threshold:
                mx1, my1, mx2, my2 = m
                merged[i] = (min(rx1, mx1), min(ry1, my1), max(rx2, mx2), max(ry2, my2))
                did_merge = True
                break
        if not did_merge:
            merged.append(r)

    # Second pass to collapse any transitive overlaps.
    if len(merged) == len(rois):
        return merged
    return _merge_rois(merged, iou_threshold=iou_threshold)


def _draw_roi_boxes(rgb: np.ndarray, rois: list[tuple[int, int, int, int]]) -> np.ndarray:
    from PIL import Image, ImageDraw

    img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(img)
    for (x1, y1, x2, y2) in rois:
        draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 255), width=3)
    return np.asarray(img)


def _propose_rois_from_pseudolabels(
    rgb: np.ndarray,
    proposer_model_path: str,
    proposer_imgsz: int,
    proposer_conf: float,
    device: str | None,
    roi_margin: int,
    max_rois: int,
    merge_iou: float,
    target_class_ids: list[int] | None,
) -> tuple[list[tuple[int, int, int, int]], int]:
    """Use a fast proposer YOLO pass to create ROIs (foreground mask concept)."""
    H, W = rgb.shape[:2]

    model = _load_yolo(proposer_model_path)
    bgr = rgb[:, :, ::-1]
    predict_kwargs: dict[str, Any] = {
        "imgsz": proposer_imgsz,
        "conf": proposer_conf,
        "device": device or "cpu",
        "verbose": False,
    }
    if target_class_ids:
        predict_kwargs["classes"] = target_class_ids

    res = model.predict(bgr, **predict_kwargs)[0]

    if res.boxes is None or len(res.boxes) == 0:
        return [], 0

    xyxy = res.boxes.xyxy.detach().cpu().numpy()
    confs = res.boxes.conf.detach().cpu().numpy() if res.boxes.conf is not None else np.ones((len(xyxy),), dtype=float)
    clss = res.boxes.cls.detach().cpu().numpy().astype(int) if res.boxes.cls is not None else np.full((len(xyxy),), -1, dtype=int)

    rois_scored: list[tuple[float, tuple[int, int, int, int]]] = []
    kept = 0
    for (x1, y1, x2, y2), score, cls_id in zip(xyxy, confs, clss, strict=False):
        if target_class_ids and cls_id not in target_class_ids:
            continue
        kept += 1
        rx1 = max(0, int(x1) - roi_margin)
        ry1 = max(0, int(y1) - roi_margin)
        rx2 = min(W, int(x2) + roi_margin)
        ry2 = min(H, int(y2) + roi_margin)
        if (rx2 - rx1) < 2 or (ry2 - ry1) < 2:
            continue
        rois_scored.append((float(score), (rx1, ry1, rx2, ry2)))

    rois_scored.sort(key=lambda t: t[0], reverse=True)
    rois = [r for _, r in rois_scored[: max_rois * 3]]  # pre-cap before merge
    rois = _merge_rois(rois, iou_threshold=merge_iou)
    rois = rois[:max_rois]
    return rois, kept


def _draw_sahi_roi_gated(
    rgb: np.ndarray,
    proposer_model_path: str,
    proposer_imgsz: int,
    proposer_conf: float,
    target_class_ids: list[int] | None,
    refiner_model_path: str,
    slice_size: int,
    overlap: float,
    conf: float,
    device: str | None,
    roi_margin: int,
    max_rois: int,
    merge_iou: float,
    nms_iou: float,
) -> tuple[np.ndarray, int, dict[str, Any]]:
    """Professor-requested optimization: run SAHI only on foreground ROIs from pseudo labels."""

    from sahi.predict import get_sliced_prediction
    from sahi.postprocess.combine import NMSPostprocess
    from sahi.utils.cv import visualize_object_predictions

    H, W = rgb.shape[:2]

    def _filter_preds_by_class_ids(object_predictions, class_ids: list[int] | None):
        if not class_ids:
            return list(object_predictions)
        allowed = set(int(x) for x in class_ids)
        out = []
        for op in object_predictions:
            cat = getattr(op, "category", None)
            cid = getattr(cat, "id", None) if cat is not None else None
            if cid is None:
                continue
            if int(cid) in allowed:
                out.append(op)
        return out

    t0 = time.perf_counter()
    rois, proposer_kept = _propose_rois_from_pseudolabels(
        rgb,
        proposer_model_path=proposer_model_path,
        proposer_imgsz=proposer_imgsz,
        proposer_conf=proposer_conf,
        device=device,
        roi_margin=roi_margin,
        max_rois=max_rois,
        merge_iou=merge_iou,
        target_class_ids=target_class_ids,
    )
    t_propose_ms = (time.perf_counter() - t0) * 1000.0

    roi_area = 0
    for (x1, y1, x2, y2) in rois:
        roi_area += max(0, x2 - x1) * max(0, y2 - y1)
    fg_ratio = float(roi_area / (W * H)) if (W > 0 and H > 0) else 0.0

    resolved_refiner = _resolve_model_path_for_sahi(refiner_model_path)
    detection_model = _load_sahi_detection_model(
        model_type="yolov8",
        model_path=resolved_refiner,
        confidence_threshold=float(conf),
        device=device or "cpu",
    )

    all_preds = []
    t1 = time.perf_counter()

    if not rois:
        # No foreground proposed => skip scanning background entirely.
        vis_dict = visualize_object_predictions(
            rgb,
            [],
            rect_th=None,
            text_size=None,
            text_th=None,
            hide_labels=False,
            hide_conf=False,
            output_dir=None,
        )
        vis = vis_dict["image"]
        meta = {
            "roi_count": 0,
            "rois": [],
            "foreground_area_ratio": float(fg_ratio),
            "proposer_kept": int(proposer_kept),
            "proposer_ms": float(t_propose_ms),
            "refine_ms": 0.0,
            "post_ms": 0.0,
        }
        return vis, 0, meta

    for (x1, y1, x2, y2) in rois:
        crop_rgb = rgb[y1:y2, x1:x2]
        # Ultralytics expects BGR when passing numpy arrays.
        crop = crop_rgb[:, :, ::-1]
        result = get_sliced_prediction(
            crop,
            detection_model,
            slice_height=slice_size,
            slice_width=slice_size,
            overlap_height_ratio=overlap,
            overlap_width_ratio=overlap,
            postprocess_type="NMS",
            perform_standard_pred=True,
            verbose=False,
        )
        for op in result.object_prediction_list:
            op.shift_amount = [int(x1), int(y1)]
            op.full_shape = [int(H), int(W)]
            all_preds.append(op.get_shifted_object_prediction())

    t_refine_ms = (time.perf_counter() - t1) * 1000.0

    t2 = time.perf_counter()
    all_preds = _filter_preds_by_class_ids(all_preds, target_class_ids)
    all_preds = NMSPostprocess(match_threshold=float(nms_iou), match_metric="IOU", class_agnostic=False)(all_preds)
    vis_dict = visualize_object_predictions(
        rgb,
        all_preds,
        rect_th=None,
        text_size=None,
        text_th=None,
        hide_labels=False,
        hide_conf=False,
        output_dir=None,
    )
    vis = vis_dict["image"]
    t_post_ms = (time.perf_counter() - t2) * 1000.0

    meta = {
        "roi_count": int(len(rois)),
        "rois": [list(r) for r in rois],
        "foreground_area_ratio": float(fg_ratio),
        "proposer_kept": int(proposer_kept),
        "proposer_ms": float(t_propose_ms),
        "refine_ms": float(t_refine_ms),
        "post_ms": float(t_post_ms),
    }
    return vis, int(len(all_preds)), meta


def _slugify(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9._-]+", "-", text)
    text = re.sub(r"-+", "-", text).strip("-")
    return text or "item"


def _now_tag() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def _load_ui_index() -> list[dict[str, Any]]:
    try:
        return json.loads(UI_INDEX_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return []
    except json.JSONDecodeError:
        return []


def _append_ui_index(entry: dict[str, Any]) -> None:
    UI_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    items = _load_ui_index()
    items.insert(0, entry)
    UI_INDEX_PATH.write_text(json.dumps(items[:500], indent=2), encoding="utf-8")


def _save_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _img_bytes_to_rgb(image_bytes: bytes) -> np.ndarray:
    from PIL import Image

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return np.array(img)


def _np_rgb_to_png_bytes(rgb: np.ndarray) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    return buf.getvalue()


def _draw_ultralytics(
    rgb: np.ndarray,
    model_path: str,
    imgsz: int,
    conf: float,
    device: str | None,
    target_class_ids: list[int] | None = None,
) -> tuple[np.ndarray, int]:
    model = _load_yolo(model_path)

    # Ultralytics uses OpenCV-style images (BGR) when passing numpy arrays.
    bgr = rgb[:, :, ::-1]
    predict_kwargs: dict[str, Any] = {"imgsz": imgsz, "conf": conf, "device": device or "cpu", "verbose": False}
    if target_class_ids:
        predict_kwargs["classes"] = target_class_ids
    res = model.predict(bgr, **predict_kwargs)[0]
    det_count = 0 if res.boxes is None else len(res.boxes)
    plotted = res.plot()  # returns BGR uint8
    bgr = plotted
    out_rgb = bgr[:, :, ::-1].copy()
    return out_rgb, det_count


def _draw_sahi(
    rgb: np.ndarray,
    model_type: str,
    model_path: str,
    slice_size: int,
    overlap: float,
    conf: float,
    device: str | None,
    target_class_ids: list[int] | None = None,
) -> tuple[np.ndarray, int]:
    from sahi.predict import get_sliced_prediction
    from sahi.utils.cv import visualize_object_predictions

    resolved = _resolve_model_path_for_sahi(model_path)
    detection_model = _load_sahi_detection_model(
        model_type=model_type,
        model_path=resolved,
        confidence_threshold=float(conf),
        device=device or "cpu",
    )

    # Ultralytics expects BGR when passing numpy arrays.
    bgr = rgb[:, :, ::-1]

    result = get_sliced_prediction(
        bgr,
        detection_model,
        slice_height=slice_size,
        slice_width=slice_size,
        overlap_height_ratio=overlap,
        overlap_width_ratio=overlap,
        postprocess_type="NMS",
        perform_standard_pred=True,
        verbose=False,
    )

    object_predictions = list(result.object_prediction_list)
    if target_class_ids:
        allowed = set(int(x) for x in target_class_ids)
        filtered = []
        for op in object_predictions:
            cat = getattr(op, "category", None)
            cid = getattr(cat, "id", None) if cat is not None else None
            if cid is not None and int(cid) in allowed:
                filtered.append(op)
        object_predictions = filtered

    # NOTE: SAHI's visualize_object_predictions returns a dict containing the annotated image.
    # It does not modify the input array in-place.
    vis_dict = visualize_object_predictions(
        rgb,
        object_predictions,
        rect_th=None,
        text_size=None,
        text_th=None,
        hide_labels=False,
        hide_conf=False,
        output_dir=None,
    )
    vis = vis_dict["image"]
    return vis, len(object_predictions)


def _video_preview_frames(video_path: Path, model_path: str, imgsz: int, conf: float, device: str, every_n: int, max_frames: int) -> list[np.ndarray]:
    """Return a list of annotated RGB frames (Ultralytics full-frame) for quick preview in the UI."""
    import cv2
    from ultralytics import YOLO

    model = YOLO(model_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    frames: list[np.ndarray] = []
    idx = 0
    kept = 0
    while kept < max_frames:
        ok, bgr = cap.read()
        if not ok:
            break
        idx += 1
        if every_n > 1 and (idx % every_n) != 0:
            continue

        res = model.predict(bgr, imgsz=imgsz, conf=conf, device=device, verbose=False)[0]
        plotted_bgr = res.plot()
        rgb = plotted_bgr[:, :, ::-1].copy()
        frames.append(rgb)
        kept += 1

    cap.release()
    return frames


def _video_preview_frames_with_log(
    video_path: Path,
    model_path: str,
    imgsz: int,
    conf: float,
    device: str,
    every_n: int,
    max_frames: int,
) -> tuple[list[np.ndarray], list[dict[str, Any]]]:
    """Return annotated RGB frames + per-frame metrics (ms, detections, frame index)."""
    import cv2
    from ultralytics import YOLO

    model = YOLO(model_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    frames: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []

    idx = 0
    kept = 0
    while kept < max_frames:
        ok, bgr = cap.read()
        if not ok:
            break
        idx += 1
        if every_n > 1 and (idx % every_n) != 0:
            continue

        t0 = time.perf_counter()
        res = model.predict(bgr, imgsz=imgsz, conf=conf, device=device, verbose=False)[0]
        ms = (time.perf_counter() - t0) * 1000.0
        det_count = 0 if res.boxes is None else len(res.boxes)

        plotted_bgr = res.plot()
        rgb = plotted_bgr[:, :, ::-1].copy()
        frames.append(rgb)
        rows.append({"frame_index": idx, "inference_ms": float(ms), "detections": int(det_count)})
        kept += 1

    cap.release()
    return frames, rows


def _detect_yolo_dataset_root(extracted_dir: Path) -> Path | None:
    """Best-effort detection of a YOLO dataset root containing images/ and labels/."""
    candidates = [extracted_dir]
    candidates.extend([p for p in extracted_dir.iterdir() if p.is_dir()])
    for c in candidates:
        if (c / "images").exists() and (c / "labels").exists():
            return c
    return None


def _infer_class_count_from_labels(labels_dir: Path) -> int | None:
    """Infer nc from YOLO txt labels by scanning class ids."""
    if not labels_dir.exists():
        return None
    max_cls = -1
    txts = list(labels_dir.rglob("*.txt"))
    for p in txts[:5000]:  # safety cap
        try:
            for line in p.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if not parts:
                    continue
                cls_id = int(float(parts[0]))
                if cls_id > max_cls:
                    max_cls = cls_id
        except Exception:
            continue
    return (max_cls + 1) if max_cls >= 0 else None


def _write_ultralytics_data_yaml(dataset_root: Path, names: list[str], train_rel: str = "images/train", val_rel: str = "images/val") -> Path:
    nc = len(names)
    yaml_text = "\n".join(
        [
            f"path: {dataset_root.as_posix()}",
            f"train: {train_rel}",
            f"val: {val_rel}",
            f"nc: {nc}",
            "names:",
        ]
        + [f"  {i}: {n}" for i, n in enumerate(names)]
        + [""]
    )
    out = dataset_root / "data.yaml"
    _write_text(out, yaml_text)
    return out


def app():
    st.set_page_config(page_title="SAHI Bench UI", layout="wide")
    _ensure_ultralytics_data_dir()

    st.title("SAHI + YOLO Bench UI")
    st.caption("Train / benchmark YOLO, run SAHI slicing, and visually inspect predictions on images or video.")

    with st.sidebar:
        page = st.radio(
            "Page",
            [
                "0) Dataset Import",
                "1) Train + Benchmark",
                "2) Visualize (Image)",
                "3) Visualize (Video/Webcam)",
                "3.5) Outputs Gallery",
                "4) Results Tables",
            ],
        )

        st.divider()
        st.write("**Workspace**")
        st.code(str(WORKSPACE_ROOT))
        st.write("**Results dir**")
        st.code(str(RESULTS_DIR))

    if page == "0) Dataset Import":
        st.subheader("Import a dataset ZIP and generate an Ultralytics data.yaml")
        st.write(
            "Upload a ZIP containing a YOLO dataset layout (recommended): `images/train`, `images/val`, `labels/train`, `labels/val`. "
            "The importer will extract it under `datasets/imports/` and generate `data.yaml` so it appears in the training dropdown."
        )

        up_zip = st.file_uploader("Dataset ZIP", type=["zip"], accept_multiple_files=False)
        dataset_name = st.text_input("Dataset name", value="my_dataset")
        classes = st.text_area(
            "Class names (one per line)",
            value="",
            help="Optional. If empty, we will try to infer nc from labels and create placeholder class names.",
        )

        if up_zip is not None and st.button("Import dataset", type="primary"):
            imports_root = WORKSPACE_ROOT / "datasets" / "imports"
            tag = _now_tag()
            dest = imports_root / f"{_slugify(dataset_name)}_{tag}"
            dest.mkdir(parents=True, exist_ok=True)

            zip_path = dest / _slugify(up_zip.name)
            zip_path.write_bytes(up_zip.getvalue())

            with st.spinner("Extracting ZIP…"):
                with zipfile.ZipFile(zip_path, "r") as zf:
                    zf.extractall(dest)

            detected_root = _detect_yolo_dataset_root(dest) or dest

            names = [c.strip() for c in classes.splitlines() if c.strip()]
            if not names:
                inferred_nc = _infer_class_count_from_labels(detected_root / "labels")
                if inferred_nc is None:
                    inferred_nc = 1
                names = [f"class{i}" for i in range(inferred_nc)]

            train_ok = (detected_root / "images" / "train").exists()
            val_ok = (detected_root / "images" / "val").exists() or (detected_root / "images" / "valid").exists()
            val_rel = "images/val" if (detected_root / "images" / "val").exists() else "images/valid"

            yaml_path = _write_ultralytics_data_yaml(detected_root, names=names, train_rel="images/train", val_rel=val_rel)

            st.success("Imported dataset and generated data.yaml")
            st.write("Dataset root:")
            st.code(str(detected_root))
            st.write("Generated YAML:")
            st.code(str(yaml_path.relative_to(WORKSPACE_ROOT)))

            if not train_ok or not val_ok:
                st.warning(
                    "Could not confirm the standard YOLO folders exist (images/train and images/val or images/valid). "
                    "You can still edit the generated data.yaml manually if needed."
                )

            st.write("Next step (example):")
            st.code(
                f"\".venv/bin/python\" -m ultralytics yolo train model=yolov8s.pt data={yaml_path.as_posix()} epochs=50 imgsz=640 device=cpu"
            )

    elif page == "1) Train + Benchmark":
        st.subheader("Train YOLO and record timing (Ultralytics vs SAHI sliced)")

        col1, col2, col3 = st.columns(3)
        with col1:
            data_yaml = st.selectbox("Dataset YAML", _find_dataset_yamls(), index=0)
            epochs = st.number_input("Epochs", min_value=1, max_value=300, value=3, step=1)
            imgsz = st.selectbox("Image size", [320, 512, 640, 768, 1024], index=2)

        with col2:
            batch = st.selectbox("Batch", [1, 2, 4, 8, 16], index=3)
            device = st.selectbox("Device", ["cpu", "mps"], index=0)
            max_infer_images = st.selectbox("Max val images for timing", [4, 8, 16, 32], index=3)

        with col3:
            yolo_weights = st.selectbox("YOLO weights", ["yolov8n.pt", "yolov8s.pt"], index=1)

        st.write("**SAHI slicing settings**")
        s1, s2, s3 = st.columns(3)
        with s1:
            slice_size = st.selectbox("Slice size", [256, 384, 512, 640, 768], index=2)
        with s2:
            overlap = st.select_slider("Overlap", options=[0.05, 0.1, 0.2, 0.3], value=0.2)
        with s3:
            conf = st.select_slider("Confidence", options=[0.01, 0.05, 0.1, 0.15, 0.25, 0.35, 0.5], value=0.25)

        args = [
            _python(),
            str(BENCHMARKS_DIR / "run_benchmark.py"),
            "--data",
            data_yaml,
            "--epochs",
            str(int(epochs)),
            "--imgsz",
            str(int(imgsz)),
            "--batch",
            str(int(batch)),
            "--device",
            device,
            "--yolo-weights",
            yolo_weights,
            "--conf",
            str(float(conf)),
            "--max-infer-images",
            str(int(max_infer_images)),
            "--sahi-slice",
            str(int(slice_size)),
            "--sahi-overlap",
            str(float(overlap)),
            "--sahi-postprocess",
            "NMS",
        ]

        st.code(" ".join(args))

        if st.button("Run benchmark", type="primary"):
            with st.spinner("Running training + benchmark (this can take a while on CPU)…"):
                rr = _run_cmd(args, cwd=WORKSPACE_ROOT)
            if rr.returncode != 0:
                st.error(f"Command failed (exit {rr.returncode})")
                if rr.stderr:
                    st.text_area("stderr", rr.stderr, height=200)
            st.success(f"Finished in {rr.seconds:.1f}s")
            if rr.stdout:
                st.text_area("stdout", rr.stdout, height=260)

            st.info("Updated tables:")
            st.write("- results/benchmark_summary.md")
            st.write("- results/<timestamp>_benchmark.json")

            latest = _latest_json("*_benchmark.json")
            if latest:
                st.write("Latest JSON:", str(latest))
                st.json(json.loads(latest.read_text(encoding="utf-8")))

    elif page == "2) Visualize (Image)":
        st.subheader("Upload an image and visualize detections")

        left, right = st.columns([1, 1])
        with left:
            up = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"], accept_multiple_files=False)
            backend = st.radio(
                "Backend",
                [
                    "Ultralytics (full image)",
                    "SAHI (sliced)",
                    "SAHI (ROI-gated via pseudo labels)",
                ],
            )

            weights = _find_weight_files()
            model_choice = st.selectbox("Model", weights, index=weights.index("yolov8s.pt") if "yolov8s.pt" in weights else 0)
            custom_model = st.text_input("Or enter model path", value="", help="Optional: workspace-relative or absolute path")
            model_path = custom_model.strip() or model_choice
            imgsz = st.selectbox("Image size", [320, 512, 640, 768, 1024], index=2)
            conf = st.select_slider("Confidence", options=[0.01, 0.05, 0.1, 0.15, 0.25, 0.35, 0.5], value=0.25)
            device = st.selectbox("Device", ["cpu", "mps"], index=0)

            slice_size = st.selectbox("Slice size (SAHI)", [256, 384, 512, 640, 768], index=2)
            overlap = st.select_slider("Overlap (SAHI)", options=[0.05, 0.1, 0.2, 0.3], value=0.2)

            if backend.startswith("SAHI (ROI-gated"):
                st.divider()
                st.write("**ROI-gated SAHI (pseudo labels)**")
                use_same_as_refiner = st.checkbox(
                    "Use the same weights as the refiner for proposer",
                    value=True,
                    help="Recommended for custom datasets. If proposer uses COCO weights but your refiner is trained on different classes, ROIs can be wrong (e.g., only one corner / wrong region).",
                )

                proposer_weights = _find_weight_files()
                if use_same_as_refiner:
                    proposer_model = model_path
                    st.caption(f"Proposer weights: {proposer_model}")
                else:
                    proposer_model = st.selectbox(
                        "Proposer model (fast)",
                        proposer_weights,
                        index=proposer_weights.index("yolov8n.pt") if "yolov8n.pt" in proposer_weights else 0,
                        help="This model proposes foreground ROIs. If ROI-gated misses objects, try a stronger proposer (e.g., yolov8s.pt or your trained weights).",
                    )

                proposer_imgsz = st.selectbox("Proposer imgsz", [256, 320, 384, 512], index=2)
                proposer_conf = st.select_slider("Proposer conf", options=[0.01, 0.05, 0.1, 0.15, 0.2, 0.25], value=0.1)
                roi_margin = st.selectbox("ROI margin (px)", [0, 16, 32, 64, 96, 128], index=3)
                max_rois = st.selectbox("Max ROIs", [1, 2, 3, 4, 6, 8], index=4)
                merge_iou = st.select_slider("ROI merge IoU", options=[0.1, 0.2, 0.3, 0.4, 0.5], value=0.3)
                nms_iou = st.select_slider("Final NMS IoU", options=[0.3, 0.4, 0.5, 0.6], value=0.5)
                st.write("**Safety net (quality)**")
                fallback_full_sahi = st.checkbox(
                    "Fallback to full SAHI if ROI-gated output is empty",
                    value=True,
                    help="If ROI-gated produces no ROIs or no detections, run full-image SAHI once. This improves recall when the proposer misses foreground.",
                )
                fallback_condition = st.selectbox(
                    "Fallback condition",
                    ["either: no ROIs OR no detections", "only: no ROIs", "only: no detections"],
                    index=0,
                )
                fallback_low_fg = st.checkbox(
                    "Also fallback if foreground coverage is too small",
                    value=True,
                    help="Recommended. If ROIs cover only a tiny part of the image (often a corner), proposer likely missed other foreground; fallback protects correctness.",
                )
                min_fg_ratio = st.select_slider(
                    "Min foreground area ratio before fallback",
                    options=[0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.1],
                    value=0.02,
                    help="If ROIs cover less than this fraction of the image, we assume proposer missed regions and run full SAHI once.",
                )

                st.write("**Debug (recommended when ROI looks wrong)**")
                show_proposer = st.checkbox(
                    "Show proposer detections overlay",
                    value=True,
                    help="Displays the proposer’s raw boxes on the full image so you can confirm it is finding the correct foreground (not only one corner).",
                )
                class_ids_text = st.text_input(
                    "Target class IDs (optional)",
                    value="",
                    help="Comma-separated integers. Example: coco car=2. Leave empty to use all classes.",
                )
                show_rois = st.checkbox("Show proposed ROIs overlay", value=True)

        with right:
            if up is None:
                st.info("Upload an image to see visual output.")
            else:
                rgb = _img_bytes_to_rgb(up.getvalue())
                st.image(rgb, caption="Input", width="stretch")

                if st.button("Run detection", type="primary"):
                    with st.spinner("Running inference…"):
                        t0 = time.perf_counter()
                        if backend.startswith("Ultralytics"):
                            out, det_count = _draw_ultralytics(
                                rgb,
                                model_path=model_path,
                                imgsz=int(imgsz),
                                conf=float(conf),
                                device=device,
                            )
                            extra_meta: dict[str, Any] = {}
                        elif backend.startswith("SAHI (ROI-gated"):
                            target_class_ids = None
                            if class_ids_text.strip():
                                try:
                                    target_class_ids = [int(x.strip()) for x in class_ids_text.split(",") if x.strip()]
                                except Exception:
                                    target_class_ids = None

                            out, det_count, extra_meta = _draw_sahi_roi_gated(
                                rgb,
                                proposer_model_path=proposer_model,
                                proposer_imgsz=int(proposer_imgsz),
                                proposer_conf=float(proposer_conf),
                                target_class_ids=target_class_ids,
                                refiner_model_path=model_path,
                                slice_size=int(slice_size),
                                overlap=float(overlap),
                                conf=float(conf),
                                device=device,
                                roi_margin=int(roi_margin),
                                max_rois=int(max_rois),
                                merge_iou=float(merge_iou),
                                nms_iou=float(nms_iou),
                            )

                            if fallback_full_sahi:
                                roi_count = int((extra_meta or {}).get("roi_count", 0))
                                fg_ratio = float((extra_meta or {}).get("foreground_area_ratio", 0.0))
                                should_fallback = False
                                if fallback_condition.startswith("either"):
                                    should_fallback = (roi_count == 0) or (int(det_count) == 0)
                                elif "no ROIs" in fallback_condition:
                                    should_fallback = (roi_count == 0)
                                else:
                                    should_fallback = (int(det_count) == 0)

                                if fallback_low_fg and (fg_ratio > 0.0) and (fg_ratio < float(min_fg_ratio)):
                                    should_fallback = True

                                if should_fallback:
                                    fb0 = time.perf_counter()
                                    out_fb, det_fb = _draw_sahi(
                                        rgb,
                                        model_type="yolov8",
                                        model_path=model_path,
                                        slice_size=int(slice_size),
                                        overlap=float(overlap),
                                        conf=float(conf),
                                        device=device,
                                        target_class_ids=target_class_ids,
                                    )
                                    fb_ms = (time.perf_counter() - fb0) * 1000.0
                                    out = out_fb
                                    det_count = det_fb
                                    extra_meta = dict(extra_meta or {})
                                    extra_meta.update(
                                        {
                                            "fallback_used": True,
                                            "fallback_backend": "SAHI (full scan)",
                                            "fallback_ms": float(fb_ms),
                                            "fallback_detections": int(det_fb),
                                        }
                                    )
                        else:
                            model_type = "yolov8"
                            out, det_count = _draw_sahi(
                                rgb,
                                model_type=model_type,
                                model_path=model_path,
                                slice_size=int(slice_size),
                                overlap=float(overlap),
                                conf=float(conf),
                                device=device,
                            )
                            extra_meta = {}
                        dt = (time.perf_counter() - t0) * 1000

                    png = _np_rgb_to_png_bytes(out)
                    st.session_state["last_image_output"] = {
                        "png": png,
                        "meta": {
                            "type": "image",
                            "input_name": up.name,
                            "backend": backend,
                            "model_path": model_path,
                            "imgsz": int(imgsz),
                            "conf": float(conf),
                            "device": device,
                            "sahi_slice": int(slice_size),
                            "sahi_overlap": float(overlap),
                            "detections": int(det_count),
                            "latency_ms": float(dt),
                            "timestamp": _now_tag(),
                            **(extra_meta or {}),
                        },
                    }

                    st.success(f"Done: {dt:.1f} ms | detections: {det_count}")
                    st.image(out, caption="Output", width="stretch")
                    st.download_button("Download PNG", data=png, file_name="prediction.png", mime="image/png")

                    if extra_meta:
                        fb_txt = ""
                        if extra_meta.get("fallback_used"):
                            fb_txt = (
                                f" | fallback={extra_meta.get('fallback_backend')}"
                                f" ({extra_meta.get('fallback_ms', 0.0):.1f} ms)"
                            )
                        st.caption(
                            f"ROI-gated breakdown: proposer={extra_meta.get('proposer_ms', 0.0):.1f} ms | "
                            f"refine={extra_meta.get('refine_ms', 0.0):.1f} ms | post={extra_meta.get('post_ms', 0.0):.1f} ms | "
                            f"rois={extra_meta.get('roi_count', 0)} | fg_area={extra_meta.get('foreground_area_ratio', 0.0):.3f}{fb_txt}"
                        )

                        if extra_meta.get("fallback_used"):
                            st.warning("ROI-gated output was empty; used fallback full-image SAHI for better recall.")

                    if backend.startswith("SAHI (ROI-gated") and show_rois and extra_meta:
                        try:
                            rois = [tuple(map(int, r)) for r in (extra_meta.get("rois") or [])]
                            st.image(
                                _draw_roi_boxes(rgb, rois),
                                caption=f"Proposed ROIs (count={len(rois)})",
                                width="stretch",
                            )
                        except Exception:
                            pass

                    if backend.startswith("SAHI (ROI-gated") and show_proposer:
                        try:
                            proposer_vis, proposer_det = _draw_ultralytics(
                                rgb,
                                model_path=proposer_model,
                                imgsz=int(proposer_imgsz),
                                conf=float(proposer_conf),
                                device=device,
                                target_class_ids=target_class_ids,
                            )
                            st.image(
                                proposer_vis,
                                caption=f"Proposer overlay (detections={proposer_det})",
                                width="stretch",
                            )
                        except Exception:
                            pass

                last = st.session_state.get("last_image_output")
                if last is not None:
                    st.divider()
                    st.write("**Save to Outputs Gallery**")
                    save_name = st.text_input("Output name", value=_slugify(Path(last["meta"]["input_name"]).stem))
                    if st.button("Save output", type="secondary"):
                        tag = last["meta"]["timestamp"]
                        out_path = UI_OUTPUTS_DIR / "images" / f"{_slugify(save_name)}_{tag}.png"
                        _save_bytes(out_path, last["png"])
                        entry = {
                            "id": f"img_{tag}_{_slugify(save_name)}",
                            "kind": "image",
                            "created_at": tag,
                            "file": str(out_path.relative_to(WORKSPACE_ROOT)),
                            "meta": last["meta"],
                        }
                        _append_ui_index(entry)
                        st.success(f"Saved: {out_path.relative_to(WORKSPACE_ROOT)}")

    elif page == "3) Visualize (Video/Webcam)":
        st.subheader("Real-time ROI detection (best for smooth, fast boxes)")
        st.write(
            "This runs the ROI pipeline script: proposer on full frame (cheap) + refiner on ROIs (fast + stable), "
            "which avoids SAHI tile-by-tile scanning artifacts."
        )

        up_vid = st.file_uploader("Upload video (optional)", type=["mp4", "mov", "avi", "mkv"], accept_multiple_files=False)
        source = st.text_input("Source", value="0", help="Webcam index (0) or a path to a video file (ignored if you upload a video)")
        proposer = st.selectbox("Proposer", ["yolov8n.pt", "yolov8s.pt"], index=0)
        refiner = st.selectbox("Refiner", ["yolov8n.pt", "yolov8s.pt"], index=1)
        proposer_imgsz = st.selectbox("Proposer imgsz", [256, 320, 384, 512], index=1)
        refiner_imgsz = st.selectbox("Refiner imgsz", [512, 640, 768], index=1)
        refresh_every = st.selectbox("Refresh every N frames", [1, 5, 10, 15, 20], index=2)
        max_rois = st.selectbox("Max ROIs", [1, 2, 4, 6, 8], index=3)
        roi_margin = st.selectbox("ROI margin (px)", [16, 32, 64, 96, 128], index=2)
        device = st.selectbox("Device", ["cpu", "mps"], index=0)

        st.divider()
        st.write("**Quick in-browser preview (full-frame YOLO)**")
        preview_model = st.selectbox("Preview model", _find_weight_files(), index=0)
        preview_imgsz = st.selectbox("Preview imgsz", [320, 512, 640, 768], index=2)
        preview_conf = st.select_slider("Preview conf", options=[0.01, 0.05, 0.1, 0.15, 0.25, 0.35, 0.5], value=0.25)
        every_n = st.selectbox("Preview every Nth frame", [1, 2, 5, 10], index=2)
        max_frames = st.selectbox("Max preview frames", [3, 5, 10, 15], index=1)
        enable_log = st.checkbox("Log per-frame timings (CSV)", value=True)

        cmd = [
            _python(),
            str(BENCHMARKS_DIR / "realtime_roi_video.py"),
            "--source",
            source,
            "--show",
            "--device",
            device,
            "--proposer",
            proposer,
            "--proposer-imgsz",
            str(int(proposer_imgsz)),
            "--refiner",
            refiner,
            "--refiner-imgsz",
            str(int(refiner_imgsz)),
            "--refresh-every",
            str(int(refresh_every)),
            "--max-rois",
            str(int(max_rois)),
            "--roi-margin",
            str(int(roi_margin)),
        ]

        st.code(" ".join(cmd))
        st.warning("This opens an OpenCV window. Press 'q' to quit.")

        headless = (
            (os.environ.get("DISPLAY") in (None, ""))
            and (os.environ.get("WAYLAND_DISPLAY") in (None, ""))
            and sys.platform.startswith("linux")
        )

        if headless:
            st.info("OpenCV GUI windows are disabled in headless deployments. Use the in-browser preview below.")

        if st.button("Launch video window", type="primary", disabled=headless):
            if up_vid is not None:
                tmp = Path(tempfile.mkdtemp(prefix="sahi_ui_vid_")) / up_vid.name
                tmp.write_bytes(up_vid.getvalue())
                cmd[cmd.index("--source") + 1] = str(tmp)
            # Use Popen so Streamlit doesn't freeze waiting forever.
            subprocess.Popen(cmd, cwd=str(WORKSPACE_ROOT))
            st.success("Launched. Look for the OpenCV window.")

        if up_vid is not None and st.button("Preview frames in UI", type="secondary"):
            tmp = Path(tempfile.mkdtemp(prefix="sahi_ui_vid_")) / up_vid.name
            tmp.write_bytes(up_vid.getvalue())
            with st.spinner("Rendering preview frames…"):
                if enable_log:
                    frames, rows = _video_preview_frames_with_log(
                        tmp,
                        model_path=preview_model,
                        imgsz=int(preview_imgsz),
                        conf=float(preview_conf),
                        device=device,
                        every_n=int(every_n),
                        max_frames=int(max_frames),
                    )
                else:
                    frames = _video_preview_frames(
                        tmp,
                        model_path=preview_model,
                        imgsz=int(preview_imgsz),
                        conf=float(preview_conf),
                        device=device,
                        every_n=int(every_n),
                        max_frames=int(max_frames),
                    )
                    rows = []
            if not frames:
                st.warning("No frames decoded.")
            else:
                st.image(frames, caption=[f"frame {i}" for i in range(len(frames))], width="stretch")

                if rows:
                    ms = [r["inference_ms"] for r in rows if "inference_ms" in r]
                    if ms:
                        st.success(
                            f"Preview stats: mean {float(np.mean(ms)):.1f} ms | p50 {float(np.median(ms)):.1f} ms | "
                            f"approx FPS {1000.0 / float(np.mean(ms)):.1f}"
                        )

                    csv_lines = ["frame_index,inference_ms,detections"]
                    csv_lines += [f"{r['frame_index']},{r['inference_ms']},{r['detections']}" for r in rows]
                    csv_bytes = ("\n".join(csv_lines) + "\n").encode("utf-8")
                    st.download_button(
                        "Download CSV log",
                        data=csv_bytes,
                        file_name=f"video_preview_{_slugify(Path(up_vid.name).stem)}.csv",
                        mime="text/csv",
                    )

                    st.session_state["last_video_log"] = {
                        "csv": csv_bytes,
                        "meta": {
                            "type": "video_preview",
                            "input_name": up_vid.name,
                            "model_path": preview_model,
                            "imgsz": int(preview_imgsz),
                            "conf": float(preview_conf),
                            "device": device,
                            "every_n": int(every_n),
                            "max_frames": int(max_frames),
                            "timestamp": _now_tag(),
                        },
                    }

    elif page == "3.5) Outputs Gallery":
        st.subheader("Outputs Gallery")
        st.write("Saved outputs from the UI live here: `results/ui_outputs/`. Use this page to review and export.")

        items = _load_ui_index()
        if not items:
            st.info("No saved outputs yet. Save an image output from the Visualize (Image) page.")
        else:
            st.write(f"Saved items: {len(items)}")
            kinds = sorted({i.get("kind", "unknown") for i in items})
            kind_filter = st.multiselect("Filter by type", kinds, default=kinds)

            filtered = [i for i in items if i.get("kind", "unknown") in kind_filter]
            for entry in filtered[:50]:
                file_rel = entry.get("file")
                meta = entry.get("meta", {})
                st.write(f"**{entry.get('id','item')}**  ·  {entry.get('created_at','')}  ·  {entry.get('kind','')}")
                if file_rel:
                    path = WORKSPACE_ROOT / file_rel
                    if path.exists() and path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                        st.image(str(path), caption=file_rel, width="stretch")
                    else:
                        st.code(file_rel)
                if meta:
                    st.caption(json.dumps(meta, ensure_ascii=False))
                st.divider()

            st.write("**Export**")
            if st.button("Create ZIP of gallery", type="primary"):
                tag = _now_tag()
                zip_out = UI_OUTPUTS_DIR / f"gallery_{tag}.zip"
                with zipfile.ZipFile(zip_out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                    # Index
                    if UI_INDEX_PATH.exists():
                        zf.write(UI_INDEX_PATH, arcname="index.json")
                    # Files referenced
                    for entry in items:
                        file_rel = entry.get("file")
                        if not file_rel:
                            continue
                        p = WORKSPACE_ROOT / file_rel
                        if p.exists():
                            zf.write(p, arcname=file_rel)
                st.success("Gallery ZIP created")
                st.download_button(
                    "Download gallery ZIP",
                    data=zip_out.read_bytes(),
                    file_name=zip_out.name,
                    mime="application/zip",
                )

    else:
        st.subheader("Results tables")

        bsum = RESULTS_DIR / "benchmark_summary.md"
        rsum = RESULTS_DIR / "roi_sahi_summary.md"
        frep = RESULTS_DIR / "final_report.md"
        pref = RESULTS_DIR / "professor_algorithm_report.md"

        tabs = st.tabs(["Professor algorithm report", "Final report", "Benchmark summary", "ROI summary", "Latest JSON"])
        with tabs[0]:
            st.markdown(_read_text(pref) or "No professor_algorithm_report.md yet.")

        with tabs[1]:
            st.markdown(_read_text(frep) or "No final report found yet.")

        with tabs[2]:
            st.markdown(_read_text(bsum) or "No benchmark_summary.md yet.")

        with tabs[3]:
            st.markdown(_read_text(rsum) or "No roi_sahi_summary.md yet.")

        with tabs[4]:
            latest = _latest_json("*_benchmark.json")
            if not latest:
                st.info("No benchmark JSON found yet.")
            else:
                st.write("Latest:", str(latest))
                st.json(json.loads(latest.read_text(encoding="utf-8")))


if __name__ == "__main__":
    app()
