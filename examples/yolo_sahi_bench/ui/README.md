# UI (Streamlit)

This is a small UI to make it easy to:
- import a dataset ZIP and auto-generate an Ultralytics `data.yaml`
- run training + benchmarks from buttons
- upload an image and visually see detections (Ultralytics or SAHI sliced)
- launch the real-time ROI video pipeline (opens an OpenCV window)
- save outputs to a gallery and export them as a ZIP
- view the results tables + latest JSON

## Run

From the workspace root:

```bash
".venv/bin/python" -m streamlit run ui/streamlit_app.py
```

Then open the URL Streamlit prints (usually http://localhost:8501).

## Notes

- **Video/Webcam page** opens an OpenCV window; press `q` to quit.
- For most consistent benchmarking, use device `cpu`.
