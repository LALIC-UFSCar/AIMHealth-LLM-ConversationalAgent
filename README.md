# PersonaRAG (Reproducible)

This repository contains:
- Interactive chatbot: Streamlit + LangChain + Groq + Elasticsearch over WHO/OMS PDFs, with personas removed from chatbot runtime.
- Inference pipelines: baseline, RAG, and legacy Phase3 batch workflows.
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
- `GROQ_API_KEY`
- `GROQ_MODEL`
- `GROQ_TEMPERATURE`
- `GROQ_MAX_TOKENS`
- `PERSONARAG_DATA_DIR`
- `PERSONARAG_MODELS_DIR`
- `PERSONARAG_OUTPUT_DIR`
- `PERSONARAG_CHECKPOINT_DIR`
- `PERSONARAG_OMS_DOCS_DIR`
- `PERSONARAG_ES_HOST`
- `PERSONARAG_ES_INDEX`
- `PERSONARAG_EMBEDDING_MODEL`
- `PERSONARAG_TOP_K`
- `OPENAI_API_KEY` (only for evaluation)

## Running the Chatbot App

The chatbot no longer uses personas or local LLaMA/LoRA generation. It uses Groq through LangChain, keeps the existing Phase 3 prompt builder, and retrieves context from the `oms_mental_health_docs` Elasticsearch index.

### Step 1: Start Elasticsearch (via Docker)
Run a single-node Elasticsearch instance:

```bash
docker run -d \
  --name elasticsearch-rag \
  -p 9200:9200 \
  -e "discovery.type=single-node" \
  -e "xpack.security.enabled=false" \
  -e "ES_JAVA_OPTS=-Xms1g -Xmx1g" \
  docker.elastic.co/elasticsearch/elasticsearch:8.13.4
```

Check it:

```bash
curl http://localhost:9200
```

### Step 2: Configure `.env`

```env
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.1-8b-instant

PERSONARAG_ES_HOST=http://localhost:9200
PERSONARAG_ES_INDEX=oms_mental_health_docs
PERSONARAG_OMS_DOCS_DIR=./oms
PERSONARAG_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

GROQ_TEMPERATURE=0.7
GROQ_MAX_TOKENS=600
PERSONARAG_TOP_K=5
```

### Step 3: Build or rebuild the knowledge base

The app will create the RAG index automatically when it is missing or empty. To force a rebuild from the WHO/OMS PDFs:

```bash
python3 scripts/build_knowledge_base.py --force
```

You can also use the Streamlit sidebar button named `Rebuild knowledge base`.

### Step 4: Run the Streamlit Interface

```bash
source .venv/bin/activate
streamlit run chatbot/app.py
```

The app will be available at `http://localhost:8501`.

See `chatbot/README.md` for full chatbot setup and troubleshooting.

## Prompt architecture

Centralized in `inference/prompt_templates.py`:
- `build_phase3_prompt(persona_context, retrieved_context, user_message)`
- `build_rag_prompt(retrieved_context, user_message)`
- `build_baseline_prompt(user_message)`

The chatbot still calls `build_phase3_prompt`, passing an empty persona context. The Phase 3 system prompt itself lives in `inference/prompts/phase3_system_prompt.md`.

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
  --oms-docs-dir oms
```

### Legacy Phase3 batch (persona-aware RAG)

```bash
python -m inference.run_phase3_inference \
  --dataset data/dataset_completo_com_inferencias_final.xlsx \
  --personas inference/all_personas_dataset.json \
  --output outputs/dataset_completo_com_inferencias_final.xlsx \
  --adapter-path models/llama-2-13b-amive-esconv \
  --es-host http://localhost:9200 \
  --es-index oms_mental_health_docs \
  --oms-docs-dir oms
```

The script will populate the RAG index automatically if `oms_mental_health_docs` is empty, using the PDFs configured in `PERSONARAG_OMS_DOCS_DIR` or the repo-local `oms/` directory.

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
