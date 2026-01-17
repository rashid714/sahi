"""Streamlit Cloud entrypoint.

Streamlit Community Cloud sometimes mis-detects nested app paths during setup.
This thin wrapper guarantees a stable root-level script entry.
"""

from __future__ import annotations

import pathlib
import runpy


def _main() -> None:
    app_path = pathlib.Path(__file__).resolve().parent / "ui" / "streamlit_app.py"
    runpy.run_path(str(app_path), run_name="__main__")


if __name__ == "__main__":
    _main()
