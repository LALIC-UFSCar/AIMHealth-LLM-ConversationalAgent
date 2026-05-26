# Fine-Tuning: Adaptação de LLaMA-3 no dataset ESConv

Este diretório contém o pipeline de treinamento reprodutível (*fine-tuning*) responsável por sintonizar o modelo base (`meta-llama/Meta-Llama-3-8B-Instruct`) em um Agente de Suporte Emocional. O processo é realizado sobre o dataset ESConv (Emotional Support Conversation).

## 🌊 Fluxo Completo de Funcionamento

O fluxo utiliza QLoRA (Quantized Low-Rank Adaptation) em 4 bits e segue as etapas abaixo:

### 1. Construção e Processamento do Dataset
- **Download / Carregamento:** Usa a biblioteca `datasets` da Hugging Face para consumir nativamente o dataset `thu-coai/esconv`.
- **Formatação de Diálogos (`format_one_dialogue_to_examples`):** Constrói amostras iterativas de diálogos. Cada resposta do usuário gera uma respectiva saída do sistema mapeada para o formato "Instruct" do LLaMA-3 (tags como `<|start_header_id|>`).
- **Histórico Acumulativo (Expanding Window):** Se a flag de janela expansiva estiver ativa (`--use-expanding-window`), todo o histórico linear da conversa é compactado dentro do prompt do turno, ajudando a dar uma longa continuidade psicológica na empatia gerada.
- **Prompt Base e Few-Shot Constraints:** Aplica regras absolutas ao prompt de treinamento definindo que o Bot **não deve relatar emoções/memórias como humano** e que seu foco não é terapia profissional de diagnóstico, mas apenas **acolhimento empático**.

### 2. Preparação do Modelo e Quantização (PEFT & QLoRA)
- Carrega o peso gigante do LLaMA-3 nativamente em VRAM otimizada (4-bit quantizado em via `nf4` via `BitsAndBytes`).
- **LoRA Embeds (Low-Rank Adaptation):** O script atômico espalha o alvo do adaptador esparso sob todas as grandes camadas de decisão densa e atenção (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`).

### 3. Loop de Treinamento Controlado
- O motor utiliza o `SFTTrainer` (Supervised Fine-tuning Trainer) provido pelas bibs de treinamento otimizadas, focando apenas nos turnos de conversas processadas.
- Aplica estratégias seguras para grandes modelos como: Ponto de verificação de gradiente (`gradient_checkpointing`), otimização baseada em *AdamW* e escalonamento em formato de senoide (*cosine lr scheduler*).

### 4. Empacotamento das Evidências (Artifacts)
- Ao fim do fine-tuning do motor local, o script salva seus **Pesos de Adaptação (Adapter)** e os *tokens* na pasta de destino dinâmico.
- Além do peso, são armazenados os metadados brutos que levaram a ele (`training_config.json` e `training_prompt_template.txt`), facilitando o versionamento para que a equipe saiba sempre os hiperparâmetros de base deste deploy.

---

## 🚀 Como Usar e Executar

Siga os passos abaixo para realizar o fine-tuning.

### Pré-requisitos
1. Uma **GPU** com memória VRAM compatível para o fine-tuning de 8 Bilhões de Parâmetros (Mesmo em QLoRA, ideal a partir de 16GB VRAM a 24GB VRAM).
2. Login ativo da [HuggingFace CLI](https://huggingface.co/docs/huggingface_hub/en/guides/cli) local que autorize baixar a arquitetura base do LLaMA-3. (Comando: `huggingface-cli login`).

### Executando via Linha de Comando:

Estando no diretório raiz do banco (`personarag_github`), execute o script como um módulo Python:

```bash
python -m finetunning.finetune_llama3_esconv \
    --base-model "meta-llama/Meta-Llama-3-8B-Instruct" \
    --dataset-name "thu-coai/esconv" \
    --num-train-epochs 3 \
    --learning-rate 2e-4
```

### Argumentos Principais Disponíveis

O pipeline inteiro pode ser tunado nos hiperparâmetros. Os essenciais incluem:

| Argumento | Descrição | Valor Default |
|---|---|---|
| `--base-model` | Repositório base do modelo da HuggingFace | `meta-llama/Meta-Llama-3-8B-Instruct` |
| `--dataset-name` | Caminho do dataset no grid da HuggingFace | `thu-coai/esconv` |
| `--output-dir` | Local físico onde sairão os adaptadores terminados | `$PERSONARAG_FINETUNE_OUTPUT_DIR/llama3-esconv-...` |
| `--num-train-epochs` | Rodadas do loop de treino sobre a base inteira | `3` |
| `--learning-rate` | Taxa de aprendizagem base (ajustada pelo scheduler) | `2e-4` |
| `--lora-r` | Tamanho matriz dos blocos de rampa fina no adaptador | `16` |
| `--lora-alpha` | Escala bruta matemática dos impulsos no adaptador | `32` |
| `--gradient-accumulation-steps` | Passos unificados para atualizar o tensor na GPU | `16` |

### O que esperar no Final da Execução?
Seu console exibirá a barra de progresso do `SFTTrainer` (tempo variável conforme hardware).
Ao finalizar, você obterá uma sub-pasta na formatação `llama3-esconv-{YYYYMMDD_HHMMSS}` salva dentro da variável de pasta final. Este destino vai portar os arquivos essenciais (`adapter_model.safetensors`, `adapter_config.json`, tokenizadores, json metadados da config.) – que posteriormente alimentam a inicialização da Fase 3 de inferência RAG.