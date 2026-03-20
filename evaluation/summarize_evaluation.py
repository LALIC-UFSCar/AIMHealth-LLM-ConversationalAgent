#!/usr/bin/env python3
"""CLI to summarize LLM evaluation results from CSV/JSON/JSONL."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from config import DEFAULT_OUTPUT_DIR, init_env, path_from_env
from evaluation.io import save_dataframe
from evaluation.scoring import build_summary_table


def parse_args() -> argparse.Namespace:
    init_env()
    output_dir = path_from_env("PERSONARAG_OUTPUT_DIR", DEFAULT_OUTPUT_DIR)
    parser = argparse.ArgumentParser(description="Summarize evaluation metrics")
    parser.add_argument("--input", type=Path, required=True, help="Evaluation results file (CSV/JSON/JSONL)")
    parser.add_argument("--output", type=Path, default=output_dir / "resumo_metricas_avaliacao_llm.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.input.suffix.lower() == ".csv":
        base_df = pd.read_csv(args.input)
    elif args.input.suffix.lower() in {".jsonl", ".ndjson"}:
        base_df = pd.read_json(args.input, lines=True)
    elif args.input.suffix.lower() == ".json":
        base_df = pd.read_json(args.input)
    else:
        raise ValueError(f"Unsupported input format: {args.input.suffix}")

    summary_df = build_summary_table(base_df)
    save_dataframe(summary_df, args.output)
    print(f"Rows in summary: {len(summary_df)}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
