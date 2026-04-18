# history_manager.py

import os
import json
from datetime import datetime
import pandas as pd

HISTORY_DIR = "cleaned_versions"
HISTORY_FILE = os.path.join(HISTORY_DIR, "history.json")

# Make sure storage exists
os.makedirs(HISTORY_DIR, exist_ok=True)
if not os.path.exists(HISTORY_FILE):
    with open(HISTORY_FILE, "w") as f:
        json.dump([], f)

def save_cleaning_session(version_id: str, df: pd.DataFrame, code: str, instructions: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Save CSV and Python script
    csv_path = os.path.join(HISTORY_DIR, f"{version_id}_cleaned.csv")
    code_path = os.path.join(HISTORY_DIR, f"{version_id}_script.py")
    df.to_csv(csv_path, index=False)
    with open(code_path, "w") as f:
        f.write(code)

    # Update history log
    with open(HISTORY_FILE, "r") as f:
        history = json.load(f)

    history.append({
        "version_id": version_id,
        "timestamp": timestamp,
        "csv_path": csv_path,
        "code_path": code_path,
        "instructions": instructions
    })

    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)

def get_history():
    with open(HISTORY_FILE, "r") as f:
        return json.load(f)

def load_version(version_id: str) -> pd.DataFrame:
    csv_path = os.path.join(HISTORY_DIR, f"{version_id}_cleaned.csv")
    return pd.read_csv(csv_path)
