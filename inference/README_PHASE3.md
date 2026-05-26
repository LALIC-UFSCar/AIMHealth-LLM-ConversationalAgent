# Phase 3: Persona-Aware RAG Inference

Este documento descreve o runner `inference.run_phase3_inference`, que executa a Fase 3 do PersonaRAG: uma inferencia em lote que combina modelo LLM com adaptador LoRA, recuperacao RAG sobre documentos OMS, contexto de persona e guardrails deterministicos para mensagens de crise.

## Estado atual

As correcoes de prontidao da Fase 3 foram aplicadas:

- O argumento `--oms-docs-dir` existe e e usado para popular o indice RAG.
- O output esta padronizado em `resposta_phase3` e `resposta_phase3_debug`.
- O fallback de personas aponta para `./inference/all_personas_dataset.json`.
- O adapter padrao aponta para `./models/llama-2-13b-amive-esconv`.
- O diretorio padrao dos PDFs OMS e `./oms`.
- `requirements.txt` limita `torch` a `<2.7.0` para evitar builds CUDA 13 incompatíveis com o driver atual, e limita `elasticsearch` a `<9.0.0` para manter compatibilidade com Elasticsearch 8.x.
- O indice `oms_mental_health_docs` foi validado localmente com 1.648 chunks indexados.

O que ainda nao esta incluso no repositorio e precisa ser fornecido para a execucao completa:

- Um dataset `.xlsx` com as colunas `input` e `persona`.
- Acesso ao modelo base da Hugging Face, ou um caminho local para o modelo base ja baixado.

## Como o pipeline funciona

Para cada linha do dataset, o script executa este fluxo:

1. Carrega o dataset Excel, o JSON de personas e o modelo base com o adaptador LoRA.
2. Conecta ao Elasticsearch no indice `oms_mental_health_docs`.
3. Se o indice nao existir ou estiver vazio, le os PDFs de `--oms-docs-dir`, divide em chunks, gera embeddings com `sentence-transformers/all-MiniLM-L6-v2` e indexa os documentos.
4. Extrai apenas a fala do cliente quando a entrada tem tags como `[Client]` ou `Client:`.
5. Executa o guardrail de crise antes de qualquer chamada ao modelo. Se detectar risco, retorna uma resposta fixa com 988 e nao chama o LLM.
6. Se houver persona e a mensagem for relacionada a saude mental, aumenta a query de recuperacao com atributos seguros da persona.
7. Recupera documentos do Elasticsearch usando busca textual e vetorial.
8. Monta o prompt de Fase 3 com persona, contexto RAG e mensagem do usuario.
9. Gera a resposta com o modelo e salva o resultado no Excel final.

O indice `oms_docs`, criado por `index_docs.py`, nao e o indice usado por este runner. Ele tem outro schema (`content` em vez de `text` + `embedding`) e nao substitui `oms_mental_health_docs`.

## Setup recomendado

No diretorio raiz do repositorio:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Verifique se a GPU esta visivel:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)"
```

Resultado esperado neste servidor:

```text
2.6.0+cu124 True 12.4
```

Verifique o Elasticsearch:

```bash
curl -s http://localhost:9200
```

Se o Elasticsearch nao estiver rodando, suba uma instancia local compativel com 8.x. Exemplo:

```bash
docker run -d \
  --name elasticsearch \
  -p 9200:9200 \
  -e "discovery.type=single-node" \
  -e "xpack.security.enabled=false" \
  docker.elastic.co/elasticsearch/elasticsearch:8.15.0
```

## Comando de execucao

Use este comando a partir da raiz do repositorio:

```bash
source .venv/bin/activate

python -m inference.run_phase3_inference \
  --dataset ./data/dataset_completo_com_inferencias_final.xlsx \
  --personas ./inference/all_personas_dataset.json \
  --output ./outputs/dataset_fase3_terminado.xlsx \
  --base-model meta-llama/Llama-2-13b-chat-hf \
  --adapter-path ./models/llama-2-13b-amive-esconv \
  --es-host http://localhost:9200 \
  --es-index oms_mental_health_docs \
  --oms-docs-dir ./oms
```

Se voce ja tiver o modelo base baixado localmente, substitua `--base-model` pelo caminho local. Para o modelo `meta-llama/Llama-2-13b-chat-hf`, normalmente e necessario estar autenticado na Hugging Face e ter permissao para acessar o repositorio.

## Argumentos principais

| Argumento | Finalidade | Default atual |
|---|---|---|
| `--dataset` | Excel de entrada com as colunas `input` e `persona` | `$PERSONARAG_DATA_DIR/dataset_completo_com_inferencias_final.xlsx` |
| `--personas` | JSON com personas AMIVE e PersonaLens | `./inference/all_personas_dataset.json` se nao existir em `$PERSONARAG_DATA_DIR` |
| `--output` | Excel final com respostas e debug | `$PERSONARAG_OUTPUT_DIR/dataset_completo_com_inferencias_final.xlsx` |
| `--base-model` | Modelo base Hugging Face ou caminho local | `meta-llama/Llama-2-13b-chat-hf` |
| `--adapter-path` | Adapter LoRA fine-tuned | `$PERSONARAG_MODELS_DIR/llama-2-13b-amive-esconv` |
| `--oms-docs-dir` | PDFs OMS usados para construir o RAG | `$PERSONARAG_OMS_DOCS_DIR` ou `./oms` |
| `--es-host` | Endpoint Elasticsearch | `http://localhost:9200` |
| `--es-index` | Indice RAG usado pela Fase 3 | `oms_mental_health_docs` |
| `--embedding-model` | Modelo de embedding para indexacao e busca | `sentence-transformers/all-MiniLM-L6-v2` |
| `--top-k` | Quantidade de documentos recuperados | `5` |
| `--max-new-tokens` | Limite de tokens gerados por resposta | `600` |

## Entrada e saida

O dataset de entrada precisa ter:

- `input`: mensagem, prompt ou dialogo do usuario.
- `persona`: id da persona, por exemplo `Angela`, `Beatriz`, `Cauê`, `user51`, `user228` ou `user1309`.

O Excel final adiciona:

- `resposta_phase3`: resposta gerada pelo pipeline.
- `resposta_phase3_debug`: JSON com informacoes de debug, incluindo query limpa, uso de persona, risco de crise e caminho de guardrail.

## Verificacoes uteis

Confirmar que o CLI esta carregando:

```bash
source .venv/bin/activate
python -m inference.run_phase3_inference --help
```

Confirmar que o indice existe:

```bash
curl -s http://localhost:9200/oms_mental_health_docs/_count
```

Resultado validado localmente:

```json
{"count":1648}
```

Confirmar que as dependencias estao coerentes:

```bash
python -m pip check
```

## Problemas comuns

- `ModuleNotFoundError: No module named 'torch'`: a `.venv` nao esta ativa ou as dependencias nao foram instaladas.
- `CUDA available: False`: verifique se o PyTorch instalado e compativel com o driver NVIDIA. Neste servidor, `torch==2.6.0+cu124` funciona.
- `Elasticsearch not available`: confirme se `curl -s http://localhost:9200` responde.
- `BadRequestError(400)` no client Elasticsearch: geralmente indica client 9.x falando com servidor 8.x. Use `elasticsearch>=8.12.0,<9.0.0`.
- `FileNotFoundError` no dataset: o Excel de inferencia nao esta versionado no repositorio; passe o caminho real com `--dataset`.
- Erro de acesso ao `meta-llama/Llama-2-13b-chat-hf`: autentique na Hugging Face ou use um caminho local em `--base-model`.
