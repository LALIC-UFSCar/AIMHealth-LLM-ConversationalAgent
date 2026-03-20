#!/usr/bin/env python3
"""CLI for LLM-based evaluation without spreadsheet dependencies."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from config import DEFAULT_DATA_DIR, DEFAULT_OUTPUT_DIR, init_env, path_from_env
from evaluation.evaluator import EvaluatorConfig, evaluate_dataframe
from evaluation.io import load_records, save_dataframe
from evaluation.personas import load_personas


def parse_args() -> argparse.Namespace:
    init_env()
    data_dir = path_from_env("PERSONARAG_DATA_DIR", DEFAULT_DATA_DIR)
    output_dir = path_from_env("PERSONARAG_OUTPUT_DIR", DEFAULT_OUTPUT_DIR)

    parser = argparse.ArgumentParser(description="Evaluate responses with LLM-as-judge")
    parser.add_argument("--input", type=Path, required=True, help="CSV/JSON/JSONL file with response triplets")
    parser.add_argument("--personas", type=Path, default=data_dir / "all_personas_dataset.json")
    parser.add_argument("--output", type=Path, default=output_dir / "avaliacao_llm.csv")
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--temperature", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required in environment")

    df = load_records(args.input)
    persona_map = load_personas(args.personas)
    cfg = EvaluatorConfig(model=args.model, temperature=args.temperature)

    evaluated = evaluate_dataframe(df, persona_map, cfg)
    save_dataframe(evaluated, args.output)
    print(f"Saved evaluation rows: {len(evaluated)}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
