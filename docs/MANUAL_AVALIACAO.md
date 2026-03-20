# Manual de Avaliacao LLM (Sem Planilhas)

Este manual descreve a avaliacao automatizada LLM-as-Judge para respostas do chatbot.

## Objetivo

Avaliar tres respostas por instancia usando os criterios:
1. Empatia
2. Adequacao ao tema
3. Personalizacao

Cada criterio recebe nota inteira de 1 a 3.

## Rubrica resumida

### Empatia
- 1: resposta fria, generica ou pouco acolhedora.
- 2: resposta acolhedora, mas superficial.
- 3: resposta calorosa, sensivel e nao julgadora.

### Adequacao ao tema
- 1: fora do tema ou com informacoes irrelevantes.
- 2: parcialmente alinhada ao tema.
- 3: alinhada ao problema central e coerente.

### Personalizacao
- 1: generica, sem conexao com contexto.
- 2: usa alguns detalhes, ainda parcialmente generica.
- 3: resposta claramente adaptada ao caso especifico.

## Formato de entrada

Arquivo CSV, JSON ou JSONL com colunas:
- persona
- id
- dataset
- input
- response_1
- response_2
- response_3

Alias tambem aceitos:
- Resposta 1 -> response_1
- Resposta 2 -> response_2
- Resposta 3 -> response_3

## Execucao

### 1) Avaliar com LLM

```bash
python -m evaluation.evaluate_llm \
  --input data/avaliacao_input.csv \
  --personas data/all_personas_dataset.json \
  --output outputs/avaliacao_llm.csv \
  --model gpt-4o
```

Requisito: variavel `OPENAI_API_KEY` definida no ambiente.

### 2) Consolidar metricas

```bash
python -m evaluation.summarize_evaluation \
  --input outputs/avaliacao_llm.csv \
  --output outputs/resumo_metricas_avaliacao_llm.csv
```

## Regras importantes

- Cada resposta deve ser avaliada de forma independente.
- Em empate de nota, use regra de desempate definida no codigo.
- Nao incluir segredos em codigo, notebook ou arquivos versionados.
- Este fluxo nao usa planilhas Excel no caminho principal.
