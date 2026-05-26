#!/usr/bin/env python3
"""Build or rebuild the Elasticsearch WHO/OMS RAG knowledge base."""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from chatbot.phase3_engine import Phase3Config, Phase3Engine


def parse_args() -> argparse.Namespace:
    defaults = Phase3Config.from_env()

    parser = argparse.ArgumentParser(description="Build the Elasticsearch mental-health RAG knowledge base from PDFs.")
    parser.add_argument("--force", action="store_true", help="Delete and rebuild the index even if it already has data.")
    parser.add_argument("--es-host", default=defaults.es_host, help="Elasticsearch host URL.")
    parser.add_argument("--es-index", default=defaults.es_index, help="Elasticsearch index name.")
    parser.add_argument("--oms-docs-dir", type=Path, default=defaults.oms_docs_dir, help="Directory containing WHO/OMS PDFs.")
    parser.add_argument("--embedding-model", default=defaults.embedding_model, help="SentenceTransformer model name.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = replace(
        Phase3Config.from_env(),
        es_host=args.es_host,
        es_index=args.es_index,
        oms_docs_dir=args.oms_docs_dir,
        embedding_model=args.embedding_model,
    )

    print("Building WHO/OMS RAG knowledge base")
    print(f"Elasticsearch host: {config.es_host}")
    print(f"Index name: {config.es_index}")
    print(f"PDF directory: {config.oms_docs_dir}")
    print(f"Embedding model: {config.embedding_model}")
    print(f"Force rebuild: {args.force}")

    engine = Phase3Engine(config)
    engine.connect_elasticsearch()
    engine.load_embedder()
    count = engine.ensure_knowledge_base(force_rebuild=args.force)

    print(f"Indexed chunks: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
