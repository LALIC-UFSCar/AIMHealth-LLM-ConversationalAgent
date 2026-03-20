# PersonaRAG (Reproducible)

This repository contains:
- Inference pipelines: baseline, RAG, and Phase3 (persona-aware RAG).
- LLM-as-judge evaluation code.
- Evaluation manual in markdown.

No spreadsheet files are required in the default workflow.

## Repository structure

- `docs/`: manuals and documentation artifacts.
- `inference/`: inference pipelines and centralized prompt templates.
- `evaluation/`: evaluation core + CLIs.
- `finetunning/`: fine-tuning scripts.
- `config.py`: shared reproducible path/env config.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Environment configuration

Copy and edit:

```bash
cp .env.example .env
```

Important variables:
- `PERSONARAG_DATA_DIR`
- `PERSONARAG_MODELS_DIR`
- `PERSONARAG_OUTPUT_DIR`
- `PERSONARAG_CHECKPOINT_DIR`
- `PERSONARAG_OMS_DOCS_DIR`
- `PERSONARAG_ES_HOST`
- `PERSONARAG_ES_INDEX`
- `OPENAI_API_KEY` (only for evaluation)

## Prompt architecture

Centralized in `inference/prompt_templates.py`:
- `build_phase3_prompt(persona_context, retrieved_context, user_message)`
- `build_rag_prompt(retrieved_context, user_message)`
- `build_baseline_prompt(user_message)`

## Inference commands

### Baseline (no RAG, no persona)

```bash
python -m inference.inferencia_baseline \
  --dataset data/Dataset_Completo.xlsx \
  --output outputs/dataset_completo_com_inferencias.xlsx \
  --adapter-path models/llama-2-amive-adapter
```

### RAG (with retrieved context, no persona)

```bash
python -m inference.inferencia_rag \
  --dataset data/dataset_completo_com_inferencias.xlsx \
  --output outputs/dataset_completo_com_inferencias_rag.xlsx \
  --oms-docs-dir data/oms_docs
```

### Phase3 (persona-aware RAG)

```bash
python -m inference.run_phase3_inference \
  --dataset data/dataset_completo_com_inferencias_final.xlsx \
  --personas data/all_personas_dataset.json \
  --output outputs/dataset_completo_com_inferencias_final.xlsx
```

## Fine-tuning (reproducible)

Script:
- `finetunning/finetune_llama3_esconv.py`

This script is reproducible and uses CLI/env paths (no hardcoded local paths).
It was adapted to use the new training prompt template requested for emotional support generation.

Example:

```bash
python -m finetunning.finetune_llama3_esconv \
  --base-model meta-llama/Meta-Llama-3-8B-Instruct \
  --dataset-name thu-coai/esconv \
  --output-dir models/finetunes/llama3-esconv-run1 \
  --num-train-epochs 3 \
  --learning-rate 2e-4
```

## Evaluation (no spreadsheets)

### 1) Score responses with LLM-as-judge

Input accepted: `.csv`, `.json`, `.jsonl` with columns:
- `persona`, `id`, `dataset`, `input`, `response_1`, `response_2`, `response_3`

```bash
python -m evaluation.evaluate_llm \
  --input data/avaliacao_input.csv \
  --personas data/all_personas_dataset.json \
  --output outputs/avaliacao_llm.csv \
  --model gpt-4o
```

### 2) Build consolidated summary

```bash
python -m evaluation.summarize_evaluation \
  --input outputs/avaliacao_llm.csv \
  --output outputs/resumo_metricas_avaliacao_llm.csv
```

See manual: `docs/MANUAL_AVALIACAO.md`.

## Security

- Do not commit `.env`.
- Do not commit API keys or tokens.
- `.gitignore` already blocks common secret/output artifacts.
