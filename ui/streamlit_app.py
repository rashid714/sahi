from __future__ import annotations

import io
import os
import tempfile
import time
import zipfile
from pathlib import Path

import numpy as np
import streamlit as st


REPO_ROOT = Path(__file__).resolve().parents[1]
DATASETS_DIR = REPO_ROOT / "datasets"
OUTPUTS_DIR = REPO_ROOT / "ui_outputs"


def _ensure_ultralytics_data_dir() -> None:
    os.environ.setdefault("ULTRALYTICS_DATA_DIR", str(DATASETS_DIR.resolve()))


def _img_bytes_to_rgb(image_bytes: bytes) -> np.ndarray:
    from PIL import Image

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return np.array(img)


def _np_rgb_to_png_bytes(rgb: np.ndarray) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    return buf.getvalue()


def _draw_ultralytics(rgb: np.ndarray, model_path: str, imgsz: int, conf: float, device: str) -> tuple[np.ndarray, int]:
    from ultralytics import YOLO

    model = YOLO(model_path)
    # Ultralytics expects BGR when passing numpy arrays
    bgr = rgb[:, :, ::-1]
    res = model.predict(bgr, imgsz=imgsz, conf=conf, device=device, verbose=False)[0]
    det_count = 0 if res.boxes is None else len(res.boxes)
    out_bgr = res.plot()
    out_rgb = out_bgr[:, :, ::-1].copy()
    return out_rgb, det_count


def _draw_sahi(rgb: np.ndarray, model_path: str, slice_size: int, overlap: float, conf: float, device: str) -> tuple[np.ndarray, int]:
    from sahi import AutoDetectionModel
    from sahi.predict import get_sliced_prediction
    from sahi.utils.cv import visualize_object_predictions

    detection_model = AutoDetectionModel.from_pretrained(
        model_type="yolov8",
        model_path=model_path,
        confidence_threshold=conf,
        device=device,
    )

    result = get_sliced_prediction(
        rgb,
        detection_model,
        slice_height=slice_size,
        slice_width=slice_size,
        overlap_height_ratio=overlap,
        overlap_width_ratio=overlap,
        postprocess_type="NMS",
        perform_standard_pred=False,
        verbose=False,
    )

    # IMPORTANT: visualize_object_predictions returns a dict (does not modify image in-place)
    vis = visualize_object_predictions(rgb, result.object_prediction_list, output_dir=None)["image"]
    return vis, len(result.object_prediction_list)


def _video_preview_frames(video_path: Path, model_path: str, imgsz: int, conf: float, device: str, every_n: int, max_frames: int):
    import cv2
    from ultralytics import YOLO

    model = YOLO(model_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    frames = []
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
        frames.append(plotted_bgr[:, :, ::-1].copy())
        kept += 1

    cap.release()
    return frames


def page_dataset_import():
    st.header("Dataset Import (optional)")
    st.write("Upload a YOLO-format dataset ZIP. It will be extracted under `datasets/imports/`.")

    up = st.file_uploader("Dataset ZIP", type=["zip"], accept_multiple_files=False)
    name = st.text_input("Dataset name", value="my_dataset")

    if up is None:
        return

    if st.button("Import", type="primary"):
        dest = DATASETS_DIR / "imports" / f"{name}_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
        dest.mkdir(parents=True, exist_ok=True)
        zip_path = dest / up.name
        zip_path.write_bytes(up.getvalue())
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(dest)
        st.success("Imported")
        st.code(str(dest))


def page_image():
    st.header("Visualize (Image)")
    left, right = st.columns([1, 1])

    with left:
        up = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"], accept_multiple_files=False)
        backend = st.radio("Backend", ["Ultralytics (full image)", "SAHI (sliced)"])
        model_path = st.selectbox("Model", ["yolov8n.pt", "yolov8s.pt"], index=1)
        imgsz = st.selectbox("Image size", [320, 512, 640, 768, 1024], index=2)
        conf = st.slider("Confidence", min_value=0.01, max_value=0.90, value=0.25, step=0.01)
        device = st.selectbox("Device", ["cpu", "mps"], index=0)
        slice_size = st.selectbox("Slice size (SAHI)", [256, 384, 512, 640, 768], index=2)
        overlap = st.selectbox("Overlap (SAHI)", [0.05, 0.1, 0.2, 0.3], index=2)

    with right:
        if up is None:
            st.info("Upload an image.")
            return

        rgb = _img_bytes_to_rgb(up.getvalue())
        st.image(rgb, caption="Input", use_container_width=True)

        if st.button("Run detection", type="primary"):
            with st.spinner("Running inference…"):
                t0 = time.perf_counter()
                if backend.startswith("Ultralytics"):
                    out, det = _draw_ultralytics(rgb, model_path=model_path, imgsz=int(imgsz), conf=float(conf), device=device)
                else:
                    out, det = _draw_sahi(rgb, model_path=model_path, slice_size=int(slice_size), overlap=float(overlap), conf=float(conf), device=device)
                ms = (time.perf_counter() - t0) * 1000.0

            st.success(f"Done: {ms:.1f} ms | detections: {det}")
            st.image(out, caption="Output", use_container_width=True)

            png = _np_rgb_to_png_bytes(out)
            st.download_button("Download PNG", data=png, file_name="prediction.png", mime="image/png")

            OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
            out_path = OUTPUTS_DIR / f"{Path(up.name).stem}_{int(time.time())}.png"
            out_path.write_bytes(png)
            st.caption(f"Saved locally to: {out_path}")


def page_video():
    st.header("Visualize (Video) — preview frames")
    st.write("On Streamlit Cloud you can preview frames in-browser. (OpenCV windows do not work in the cloud.)")

    up = st.file_uploader("Upload video", type=["mp4", "mov", "avi", "mkv"], accept_multiple_files=False)
    model_path = st.selectbox("Model", ["yolov8n.pt", "yolov8s.pt"], index=0)
    imgsz = st.selectbox("Image size", [320, 512, 640, 768], index=2)
    conf = st.slider("Confidence", min_value=0.01, max_value=0.90, value=0.25, step=0.01)
    device = st.selectbox("Device", ["cpu", "mps"], index=0)
    every_n = st.selectbox("Preview every Nth frame", [1, 2, 5, 10], index=2)
    max_frames = st.selectbox("Max preview frames", [3, 5, 10, 15], index=1)

    if up is None:
        return

    if st.button("Preview", type="primary"):
        tmp = Path(tempfile.mkdtemp(prefix="sahi_ui_vid_")) / up.name
        tmp.write_bytes(up.getvalue())
        with st.spinner("Decoding + running YOLO…"):
            frames = _video_preview_frames(tmp, model_path=model_path, imgsz=int(imgsz), conf=float(conf), device=device, every_n=int(every_n), max_frames=int(max_frames))
        if not frames:
            st.warning("No frames decoded.")
        else:
            st.image(frames, caption=[f"frame {i}" for i in range(len(frames))], use_container_width=True)


def main():
    st.set_page_config(page_title="SAHI + YOLO UI", layout="wide")
    _ensure_ultralytics_data_dir()

    st.title("SAHI + YOLO (Streamlit)")
    st.caption("Image + video preview using Ultralytics YOLO and SAHI sliced inference.")

    page = st.sidebar.radio("Page", ["Dataset Import", "Image", "Video"])

    if page == "Dataset Import":
        page_dataset_import()
    elif page == "Image":
        page_image()
    else:
        page_video()


if __name__ == "__main__":
    main()
