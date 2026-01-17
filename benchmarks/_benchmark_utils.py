from __future__ import annotations

import json
import os
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def workspace_root() -> Path:
    return Path(__file__).resolve().parents[1]


def ensure_ultralytics_data_dir() -> Path:
    # Keep datasets under workspace/datasets for predictable paths and to match the Streamlit UI.
    root = workspace_root() / "datasets"
    root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("ULTRALYTICS_DATA_DIR", str(root.resolve()))
    return root


def now_utc_compact() -> str:
    # Avoid ':' for macOS filenames
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def detect_device() -> str:
    # Prefer MPS on Apple Silicon when available.
    try:
        import torch

        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def system_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "python": sys.version.replace("\n", " "),
        "executable": sys.executable,
    }
    try:
        import torch

        info.update(
            {
                "torch_version": getattr(torch, "__version__", None),
                "mps_available": bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()),
            }
        )
    except Exception as e:
        info["torch_error"] = repr(e)
    return info


@dataclass
class TimedResult:
    seconds: float
    value: Any


def timed(fn, *args, **kwargs) -> TimedResult:
    start = time.perf_counter()
    value = fn(*args, **kwargs)
    end = time.perf_counter()
    return TimedResult(seconds=end - start, value=value)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def append_markdown_row(path: Path, header: list[str], row: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(
            "| " + " | ".join(header) + " |\n" + "| " + " | ".join(["---"] * len(header)) + " |\n",
            encoding="utf-8",
        )
    with path.open("a", encoding="utf-8") as f:
        f.write("| " + " | ".join(row) + " |\n")


def safe_float(v: Any) -> float | None:
    try:
        if v is None:
            return None
        return float(v)
    except Exception:
        return None
