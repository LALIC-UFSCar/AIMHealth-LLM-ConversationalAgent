# Mental Health RAG Chatbot

## Overview

This Streamlit chatbot uses LangChain, Groq, Elasticsearch, and WHO/OMS PDFs to generate grounded mental-health support responses.

The runtime flow is:

1. Clean the user message.
2. Run the deterministic crisis guardrail from `inference.guardrail`.
3. Retrieve WHO/OMS context from Elasticsearch using hybrid BM25 plus dense-vector search.
4. Build the existing Phase 3 prompt with `inference.prompt_templates.build_phase3_prompt`.
5. Send the prompt to Groq through `langchain-groq`.

Personas have been removed from the chatbot runtime. The app no longer loads persona files, augments retrieval with persona metadata, injects persona context into prompts, or exposes persona selection in the UI. Local LLaMA, LoRA, CUDA, Transformers, PEFT, and bitsandbytes are not used for chatbot generation.

## Requirements

- Python 3.10 or newer.
- Elasticsearch 8.x running locally or remotely.
- A Groq API key.
- WHO/OMS PDFs in the configured docs directory. The repo default is `./oms`.
- Python dependencies from `requirements.txt`, including:
  - `streamlit`
  - `langchain`
  - `langchain-core`
  - `langchain-groq`
  - `elasticsearch`
  - `sentence-transformers`
  - `pypdf`
  - `python-dotenv`

Install dependencies from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Environment Setup

Create `.env` in the repository root:

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

`PERSONARAG_OMS_DOCS_DIR` may point to any directory containing `.pdf` files. The default remains compatible with this repo's `./oms` directory.

## Starting Elasticsearch

Run a local single-node Elasticsearch container:

```bash
docker run -d \
  --name elasticsearch-rag \
  -p 9200:9200 \
  -e "discovery.type=single-node" \
  -e "xpack.security.enabled=false" \
  -e "ES_JAVA_OPTS=-Xms1g -Xmx1g" \
  docker.elastic.co/elasticsearch/elasticsearch:8.13.4
```

Check that it is available:

```bash
curl http://localhost:9200
```

If you already have Elasticsearch running elsewhere, set `PERSONARAG_ES_HOST` to that URL.

## Building Or Rebuilding The Knowledge Base

The Elasticsearch index uses this schema:

- `chunk_id`
- `source`
- `chunk_index`
- `text`
- `char_count`
- `embedding`

The default index name is `oms_mental_health_docs`. The engine creates the index automatically if it is missing or empty.

You can rebuild it from Streamlit with the sidebar button:

```text
Rebuild knowledge base
```

You can also rebuild it from the command line:

```bash
python3 scripts/build_knowledge_base.py --force
```

The CLI loads `.env`, connects to Elasticsearch, extracts text from the WHO/OMS PDFs, chunks with the existing defaults, embeds with SentenceTransformer, recreates the index when `--force` is used, and prints the indexed chunk count.

Default build without deleting an existing populated index:

```bash
python3 scripts/build_knowledge_base.py
```

Override settings if needed:

```bash
python3 scripts/build_knowledge_base.py \
  --es-host http://localhost:9200 \
  --es-index oms_mental_health_docs \
  --oms-docs-dir ./oms \
  --embedding-model sentence-transformers/all-MiniLM-L6-v2 \
  --force
```

## Running The Chatbot

From the repository root:

```bash
source .venv/bin/activate
streamlit run chatbot/app.py
```

The app exposes:

- Groq model.
- Elasticsearch host.
- RAG index.
- WHO/OMS PDFs directory.
- Embedding model.
- Top K.
- Max tokens.
- Temperature.
- Knowledge-base rebuild button.
- Retrieved document and debug expanders.
- Clear conversation button.

Debug output includes the cleaned query, crisis status, retrieved documents, Elasticsearch index, Groq model, and top-k.

## Troubleshooting

`GROQ_API_KEY is required`

Set `GROQ_API_KEY` in `.env`, then restart Streamlit.

`Elasticsearch is not available`

Start Elasticsearch and verify it with `curl http://localhost:9200`. If it is remote, update `PERSONARAG_ES_HOST`.

`No PDF files found`

Check `PERSONARAG_OMS_DOCS_DIR`. It must point to a directory containing `.pdf` files.

`RAG docs` is zero

Use the Streamlit `Rebuild knowledge base` button or run `python3 scripts/build_knowledge_base.py --force`.

Invalid Groq model name

Set `GROQ_MODEL` to a Groq chat model available for your account, then restart Streamlit.

Dependency installation problems

Upgrade `pip`, reinstall from `requirements.txt`, and make sure the virtual environment is active:

```bash
python3 -m pip install --upgrade pip
pip install -r requirements.txt
```
