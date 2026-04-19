# history_manager.py

import os
import re
import json
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

HISTORY_DIR = "cleaned_versions"
HISTORY_FILE = os.path.join(HISTORY_DIR, "history.json")
ANALYSIS_HISTORY_FILE = os.path.join(HISTORY_DIR, "analysis_history.json")

# Make sure storage exists
os.makedirs(HISTORY_DIR, exist_ok=True)
if not os.path.exists(HISTORY_FILE):
    with open(HISTORY_FILE, "w") as f:
        json.dump([], f)
if not os.path.exists(ANALYSIS_HISTORY_FILE):
    with open(ANALYSIS_HISTORY_FILE, "w") as f:
        json.dump([], f)


def _read_json_list(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data if isinstance(data, list) else []


def _write_json_list(path: str, data: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _json_safe(value: Any) -> Any:
    """Convert common pandas/numpy values into JSON-safe objects."""
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="records")
    if isinstance(value, pd.Series):
        return value.to_dict()
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()

    # Numpy scalar/array fallback without importing numpy directly.
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            pass

    if isinstance(value, (str, int, float, bool)) or value is None:
        return value

    return str(value)


def get_next_version_id() -> str:
    history = get_history()
    max_idx = 0

    for item in history:
        version_id = item.get("version_id", "")
        match = re.match(r"^v(\d+)$", version_id)
        if match:
            max_idx = max(max_idx, int(match.group(1)))

    return f"v{max_idx + 1}"


def save_cleaning_session(
    version_id: str,
    df: pd.DataFrame,
    code: str,
    instructions: str,
    source_name: Optional[str] = None,
):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Save CSV and Python script
    csv_path = os.path.join(HISTORY_DIR, f"{version_id}_cleaned.csv")
    code_path = os.path.join(HISTORY_DIR, f"{version_id}_script.py")
    df.to_csv(csv_path, index=False)
    with open(code_path, "w") as f:
        f.write(code)

    # Update history log
    history = _read_json_list(HISTORY_FILE)

    history.append(
        {
            "version_id": version_id,
            "timestamp": timestamp,
            "csv_path": csv_path,
            "code_path": code_path,
            "instructions": instructions,
            "source_name": source_name or "",
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
        }
    )

    _write_json_list(HISTORY_FILE, history)


def get_history() -> List[Dict[str, Any]]:
    return _read_json_list(HISTORY_FILE)


def load_version(version_id: str) -> pd.DataFrame:
    csv_path = os.path.join(HISTORY_DIR, f"{version_id}_cleaned.csv")
    return pd.read_csv(csv_path)


def save_analysis_session(
    version_id: str,
    analysis_type: str,
    result: Dict[str, Any],
    notes: str = "",
) -> str:
    """Persist a completed analysis run so it survives refresh/restart."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    file_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_type = re.sub(r"[^a-z0-9]+", "_", analysis_type.lower()).strip("_")
    analysis_id = f"a{int(datetime.now().timestamp())}"

    result_path = os.path.join(HISTORY_DIR, f"{version_id}_{safe_type}_{file_stamp}.json")
    payload = {
        "analysis_id": analysis_id,
        "version_id": version_id,
        "analysis_type": analysis_type,
        "timestamp": timestamp,
        "result": _json_safe(result),
        "notes": notes,
    }

    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    history = _read_json_list(ANALYSIS_HISTORY_FILE)
    history.append(
        {
            "analysis_id": analysis_id,
            "version_id": version_id,
            "analysis_type": analysis_type,
            "timestamp": timestamp,
            "result_path": result_path,
            "notes": notes,
        }
    )
    _write_json_list(ANALYSIS_HISTORY_FILE, history)
    return analysis_id


def get_analysis_history(version_id: Optional[str] = None) -> List[Dict[str, Any]]:
    history = _read_json_list(ANALYSIS_HISTORY_FILE)
    if version_id:
        history = [item for item in history if item.get("version_id") == version_id]
    return sorted(history, key=lambda item: item.get("timestamp", ""), reverse=True)


def load_analysis_session(analysis_id: str) -> Dict[str, Any]:
    history = _read_json_list(ANALYSIS_HISTORY_FILE)
    match = next((item for item in history if item.get("analysis_id") == analysis_id), None)
    if not match:
        raise ValueError(f"Analysis session not found: {analysis_id}")

    result_path = match.get("result_path", "")
    if not result_path or not os.path.exists(result_path):
        raise FileNotFoundError(f"Saved analysis result file not found for {analysis_id}")

    with open(result_path, "r", encoding="utf-8") as f:
        return json.load(f)
