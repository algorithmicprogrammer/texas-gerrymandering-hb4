import json
import os
from typing import Any, Dict

import pandas as pd

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def read_table(path: str) -> pd.DataFrame:
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    if path.endswith(".csv"):
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file type: {path}")

def write_parquet(df: pd.DataFrame, path: str) -> None:
    ensure_dir(os.path.dirname(path))
    df.to_parquet(path, index=False)

def write_json(obj: Dict[str, Any], path: str) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
