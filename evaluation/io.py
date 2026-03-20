"""I/O utilities for evaluation in CSV/JSONL formats."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = [
    "persona",
    "id",
    "dataset",
    "input",
    "response_1",
    "response_2",
    "response_3",
]

ALIAS_COLUMNS = {
    "Resposta 1": "response_1",
    "Resposta 2": "response_2",
    "Resposta 3": "response_3",
}


def load_records(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    elif path.suffix.lower() in {".jsonl", ".ndjson"}:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        df = pd.DataFrame(rows)
    elif path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and "records" in payload:
            payload = payload["records"]
        df = pd.DataFrame(payload)
    else:
        raise ValueError(f"Unsupported input format: {path.suffix}")

    df = df.rename(columns=ALIAS_COLUMNS)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return df


def save_dataframe(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".csv":
        df.to_csv(path, index=False)
        return
    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        with path.open("w", encoding="utf-8") as f:
            for _, row in df.iterrows():
                f.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")
        return
    if path.suffix.lower() == ".json":
        path.write_text(json.dumps(df.to_dict(orient="records"), ensure_ascii=False, indent=2), encoding="utf-8")
        return
    raise ValueError(f"Unsupported output format: {path.suffix}")
