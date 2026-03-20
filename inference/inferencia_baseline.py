#!/usr/bin/env python3
"""
Script de Inferência Baseline (Sem RAG)
Objetivo: Gerar respostas para todas as 240 instâncias do dataset_completo.xlsx
Modelo: LLaMA-2-7B-Chat + LoRA ESConv (mesmo usado no notebook inferencia_sem_rag.ipynb)
"""

import pandas as pd
import torch
import os
import argparse
from pathlib import Path
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm
from inference.prompt_templates import build_baseline_prompt
from config import (
    DEFAULT_CHECKPOINT_DIR,
    DEFAULT_DATA_DIR,
    DEFAULT_MODELS_DIR,
    DEFAULT_OUTPUT_DIR,
    ensure_parent_dir,
    init_env,
    path_from_env,
)

# ============================================================================
# CONFIGURAÇÕES (Mesmas do notebook inferencia_sem_rag.ipynb)
# ============================================================================

BASE_MODEL = "meta-llama/Llama-2-13b-chat-hf"

# ============================================================================
# SYSTEM PROMPT (Mesmo do notebook rag_elasticsearch_oms_inferencia.ipynb)
# Para garantir comparação justa entre baseline (sem RAG) e com RAG
# ============================================================================

def parse_args() -> argparse.Namespace:
    init_env()
    data_dir = path_from_env("PERSONARAG_DATA_DIR", DEFAULT_DATA_DIR)
    models_dir = path_from_env("PERSONARAG_MODELS_DIR", DEFAULT_MODELS_DIR)
    output_dir = path_from_env("PERSONARAG_OUTPUT_DIR", DEFAULT_OUTPUT_DIR)
    checkpoint_dir = path_from_env("PERSONARAG_CHECKPOINT_DIR", DEFAULT_CHECKPOINT_DIR)

    parser = argparse.ArgumentParser(description="Baseline inference (no RAG)")
    parser.add_argument("--dataset", type=Path, default=data_dir / "Dataset_Completo.xlsx")
    parser.add_argument("--output", type=Path, default=output_dir / "dataset_completo_com_inferencias.xlsx")
    parser.add_argument("--checkpoint", type=Path, default=checkpoint_dir / "checkpoint_baseline.xlsx")
    parser.add_argument("--adapter-path", type=Path, default=models_dir / "llama-2-amive-adapter")
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--checkpoint-every", type=int, default=20)
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 70)
    print("INFERÊNCIA BASELINE (SEM RAG)")
    print("=" * 70)
    print(f"Início: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # ========================================================================
    # PASSO 1: Verificar GPU
    # ========================================================================
    print("\n" + "=" * 70)
    print("PASSO 1: VERIFICANDO GPU")
    print("=" * 70)
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"✅ GPU: {gpu_name}")
        print(f"   Memória: {gpu_memory:.1f} GB")
    else:
        print("❌ GPU não disponível!")
        return
    
    # ========================================================================
    # PASSO 2: Carregar Modelo
    # ========================================================================
    print("\n" + "=" * 70)
    print("PASSO 2: CARREGANDO MODELO LLaMA-2-13B-Chat + LoRA Amive")
    print("=" * 70)
    
    # Quantização 4-bit
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )
    
    print(f"\n📥 Carregando tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(str(args.adapter_path))
    tokenizer.pad_token = tokenizer.eos_token
    
    print(f"📥 Carregando modelo base: {args.base_model}")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    print(f"📥 Carregando adapter LoRA...")
    model = PeftModel.from_pretrained(base_model, str(args.adapter_path))
    model.eval()
    
    print(f"\n✅ Modelo carregado com sucesso!")
    print(f"   Dispositivo: {next(model.parameters()).device}")
    
    # ========================================================================
    # PASSO 3: Definir Função de Inferência
    # ========================================================================
    print("\n" + "=" * 70)
    print("PASSO 3: CONFIGURANDO FUNÇÃO DE INFERÊNCIA")
    print("=" * 70)
    
    def generate_response(user_input):
        """
        Gera uma resposta empática usando o mesmo prompt do notebook RAG.
        """
        # Prompt centralizado para baseline (sem persona e sem RAG)
        prompt = build_baseline_prompt(user_input)
        
        # Tokenizar
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Gerar resposta
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                do_sample=True,
                top_p=args.top_p,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decodificar
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extrair apenas a resposta (depois do [/INST])
        if "[/INST]" in full_response:
            response = full_response.split("[/INST]")[-1].strip()
        else:
            response = full_response
        
        return response
    
    print("✅ Função de inferência definida!")
    print(f"   Max tokens: {args.max_new_tokens}")
    print(f"   Temperature: {args.temperature}")
    print(f"   Top-p: {args.top_p}")
    
    # ========================================================================
    # PASSO 4: Carregar Dataset
    # ========================================================================
    print("\n" + "=" * 70)
    print("PASSO 4: CARREGANDO DATASET")
    print("=" * 70)
    
    df = pd.read_excel(args.dataset)
    print(f"✅ Dataset carregado: {len(df)} instâncias")
    print(f"   Colunas: {df.columns.tolist()}")
    print(f"   Distribuição: {df['dataset'].value_counts().to_dict()}")
    
    # Verificar se já existe checkpoint
    if os.path.exists(args.checkpoint):
        df_checkpoint = pd.read_excel(args.checkpoint)
        already_done = df_checkpoint['resposta_baseline'].notna().sum()
        print(f"\n📂 Checkpoint encontrado: {already_done} instâncias já processadas")
        df = df_checkpoint
    else:
        df['resposta_baseline'] = None
        already_done = 0
    
    # ========================================================================
    # PASSO 5: Executar Inferências
    # ========================================================================
    print("\n" + "=" * 70)
    print("PASSO 5: EXECUTANDO INFERÊNCIAS")
    print("=" * 70)
    
    pending = df[df['resposta_baseline'].isna()]
    print(f"   Pendentes: {len(pending)} instâncias")
    print(f"   Tempo estimado: {len(pending) * 1.5 / 60:.1f} - {len(pending) * 3 / 60:.1f} horas")
    
    processed = 0
    errors = 0
    
    for idx, row in tqdm(pending.iterrows(), total=len(pending), desc="Inferência"):
        try:
            response = generate_response(row['input'])
            df.at[idx, 'resposta_baseline'] = response
            processed += 1
            
            # Checkpoint
            if processed % args.checkpoint_every == 0:
                ensure_parent_dir(args.checkpoint)
                df.to_excel(args.checkpoint, index=False)
                tqdm.write(f"   💾 Checkpoint salvo ({processed} processados)")
                
        except Exception as e:
            errors += 1
            df.at[idx, 'resposta_baseline'] = f"ERRO: {str(e)}"
            tqdm.write(f"   ❌ Erro no ID {row['id']}: {e}")
    
    # ========================================================================
    # PASSO 6: Salvar Resultado Final
    # ========================================================================
    print("\n" + "=" * 70)
    print("PASSO 6: SALVANDO RESULTADO")
    print("=" * 70)
    
    ensure_parent_dir(args.output)
    df.to_excel(args.output, index=False)
    
    # Remover checkpoint se existir
    if os.path.exists(args.checkpoint):
        os.remove(args.checkpoint)
    
    print(f"✅ Arquivo salvo: {args.output}")
    print(f"   Total: {len(df)} instâncias")
    print(f"   Processados: {processed}")
    print(f"   Erros: {errors}")
    
    # Estatísticas
    print("\n📊 ESTATÍSTICAS:")
    df['resposta_len'] = df['resposta_baseline'].apply(lambda x: len(str(x).split()) if pd.notna(x) else 0)
    for dataset in df['dataset'].unique():
        subset = df[df['dataset'] == dataset]
        mean_len = subset['resposta_len'].mean()
        print(f"   {dataset}: {mean_len:.0f} palavras (média)")
    
    print(f"\n✅ Finalizado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
