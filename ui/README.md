# Streamlit UI

Run locally (from repo root):

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
streamlit run ui/streamlit_app.py
```

Deploy on Streamlit Community Cloud:
- Main file path: `ui/streamlit_app.py`
- Requirements: `requirements.txt`

Notes:
- Video ROI window uses OpenCV GUI (not available on Streamlit Cloud). Use the in-browser preview there.
- For reliable CPU behavior on servers, keep device set to `cpu`.
