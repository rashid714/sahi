# Technical Report (Deployment Copy)

This repository contains a Streamlit demo UI at `ui/streamlit_app.py` that uses:
- Ultralytics YOLO (full-frame inference)
- SAHI sliced inference (`get_sliced_prediction`)

## Deploy (Streamlit Community Cloud)
- Repo: this GitHub repository
- Main file: `ui/streamlit_app.py`
- Requirements: `requirements.txt`

## Notes
- SAHI visualization uses `visualize_object_predictions(...)["image"]` (SAHI returns the annotated image; it does not mutate input in-place).
- OpenCV GUI windows do not work on Streamlit Cloud; use in-browser video preview.
