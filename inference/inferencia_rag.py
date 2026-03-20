#!/usr/bin/env python3
"""
Script de Inferência RAG (Com Elasticsearch + OMS Docs)
Objetivo: Gerar respostas com RAG para todas as 240 instâncias
Modelo: LLaMA-2-13B-Chat + LoRA Amive (mesmo do baseline)
"""

import pandas as pd
import torch
import os
import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from elasticsearch import Elasticsearch
from pypdf import PdfReader
from inference.prompt_templates import build_rag_prompt
from config import (
    DEFAULT_CHECKPOINT_DIR,
    DEFAULT_DATA_DIR,
    DEFAULT_MODELS_DIR,
    DEFAULT_OMS_DOCS_DIR,
    DEFAULT_OUTPUT_DIR,
    ensure_parent_dir,
    init_env,
    path_from_env,
)

# ============================================================================
# CONFIGURAÇÕES
# ============================================================================

BASE_MODEL = "meta-llama/Llama-2-13b-chat-hf"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ============================================================================
# SYSTEM PROMPT (Mesmo do baseline + instruções RAG)
# ============================================================================

def parse_args() -> argparse.Namespace:
    init_env()
    data_dir = path_from_env("PERSONARAG_DATA_DIR", DEFAULT_DATA_DIR)
    models_dir = path_from_env("PERSONARAG_MODELS_DIR", DEFAULT_MODELS_DIR)
    output_dir = path_from_env("PERSONARAG_OUTPUT_DIR", DEFAULT_OUTPUT_DIR)
    checkpoint_dir = path_from_env("PERSONARAG_CHECKPOINT_DIR", DEFAULT_CHECKPOINT_DIR)
    oms_docs_dir = path_from_env("PERSONARAG_OMS_DOCS_DIR", DEFAULT_OMS_DOCS_DIR)

    parser = argparse.ArgumentParser(description="RAG inference with Elasticsearch")
    parser.add_argument("--dataset", type=Path, default=data_dir / "dataset_completo_com_inferencias.xlsx")
    parser.add_argument("--output", type=Path, default=output_dir / "dataset_completo_com_inferencias_rag.xlsx")
    parser.add_argument("--checkpoint", type=Path, default=checkpoint_dir / "checkpoint_rag.xlsx")
    parser.add_argument("--oms-docs-dir", type=Path, default=oms_docs_dir)
    parser.add_argument("--adapter-path", type=Path, default=models_dir / "llama-2-amive-adapter")
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--es-host", default=os.getenv("PERSONARAG_ES_HOST", "http://localhost:9200"))
    parser.add_argument("--index-name", default=os.getenv("PERSONARAG_ES_INDEX", "oms_mental_health_docs"))
    parser.add_argument("--embedding-model", default=EMBEDDING_MODEL)
    parser.add_argument("--max-new-tokens", type=int, default=350)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    parser.add_argument("--checkpoint-every", type=int, default=20)
    return parser.parse_args()


# ============================================================================
# CLASSES E FUNÇÕES
# ============================================================================

class HybridRetriever:
    """Hybrid retriever: BM25 + Dense + RRF fusion."""
    
    def __init__(self, es_client, index_name, embedder, k=5, rrf_k=60):
        self.es = es_client
        self.index_name = index_name
        self.embedder = embedder
        self.k = k
        self.rrf_k = rrf_k
    
    def _bm25_search(self, query: str, k: int) -> List[Dict]:
        """BM25 lexical search."""
        response = self.es.search(
            index=self.index_name,
            body={"query": {"match": {"text": {"query": query, "operator": "or"}}}, "size": k}
        )
        return response['hits']['hits']
    
    def _dense_search(self, query: str, k: int) -> List[Dict]:
        """Dense vector semantic search."""
        query_embedding = self.embedder.encode(query).tolist()
        response = self.es.search(
            index=self.index_name,
            body={"knn": {"field": "embedding", "query_vector": query_embedding, "k": k, "num_candidates": k * 2}}
        )
        return response['hits']['hits']
    
    def _rrf_fusion(self, bm25_results: List, dense_results: List) -> List[Dict]:
        """Reciprocal Rank Fusion."""
        scores, docs = {}, {}
        
        for rank, hit in enumerate(bm25_results):
            doc_id = hit['_id']
            scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (self.rrf_k + rank + 1)
            docs[doc_id] = hit['_source']
        
        for rank, hit in enumerate(dense_results):
            doc_id = hit['_id']
            scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (self.rrf_k + rank + 1)
            docs[doc_id] = hit['_source']
        
        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        results = []
        for doc_id, score in sorted_docs[:self.k]:
            doc = docs[doc_id]
            doc['rrf_score'] = score
            results.append(doc)
        return results
    
    def retrieve(self, query: str) -> List[Dict]:
        """Hybrid retrieval."""
        bm25 = self._bm25_search(query, self.k * 2)
        dense = self._dense_search(query, self.k * 2)
        return self._rrf_fusion(bm25, dense)


def extract_text_from_pdf(pdf_path: str) -> Dict[str, Any]:
    """Extract text from a PDF file."""
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    
    return {
        "filename": Path(pdf_path).name,
        "filepath": str(pdf_path),
        "num_pages": len(reader.pages),
        "text": text,
        "char_count": len(text)
    }


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    text_len = len(text)
    
    while start < text_len:
        end = start + chunk_size
        
        if end < text_len:
            for sep in ['. ', '.\n', '\n\n']:
                last_sep = text[start:end].rfind(sep)
                if last_sep != -1 and last_sep > chunk_size // 2:
                    end = start + last_sep + len(sep)
                    break
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        start = end - overlap
    
    return chunks


def setup_elasticsearch_index(es, embedder, embedding_dim, index_name: str, oms_docs_path: Path):
    """Create and populate Elasticsearch index if needed."""
    
    # Check if index exists and has documents
    if es.indices.exists(index=index_name):
        count = es.count(index=index_name)['count']
        if count > 0:
            print(f"   ✅ Índice '{index_name}' já existe com {count} documentos")
            return True
    
    print(f"   🔄 Criando índice '{index_name}'...")
    
    # Extract PDFs
    pdf_files = list(Path(oms_docs_path).glob("*.pdf"))
    if not pdf_files:
        print(f"   ❌ Nenhum PDF encontrado em {oms_docs_path}")
        return False
    
    documents = []
    for pdf_file in pdf_files:
        try:
            doc = extract_text_from_pdf(str(pdf_file))
            documents.append(doc)
            print(f"      📄 {pdf_file.name}: {doc['num_pages']} páginas")
        except Exception as e:
            print(f"      ❌ Erro em {pdf_file.name}: {e}")
    
    # Create chunks
    all_chunks = []
    for doc in documents:
        chunks = chunk_text(doc['text'], chunk_size=1000, overlap=200)
        for i, chunk in enumerate(chunks):
            all_chunks.append({
                'chunk_id': f"{doc['filename']}_{i}",
                'source': doc['filename'],
                'chunk_index': i,
                'text': chunk,
                'char_count': len(chunk)
            })
    
    print(f"   📝 {len(all_chunks)} chunks criados")
    
    # Generate embeddings
    print(f"   🔄 Gerando embeddings...")
    texts = [chunk['text'] for chunk in all_chunks]
    embeddings = embedder.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    
    for i, chunk in enumerate(all_chunks):
        chunk['embedding'] = embeddings[i].tolist()
    
    # Delete existing index if exists
    if es.indices.exists(index=index_name):
        es.indices.delete(index=index_name)
    
    # Create index mapping
    index_mapping = {
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0,
            "analysis": {
                "analyzer": {
                    "english_analyzer": {"type": "english"}
                }
            }
        },
        "mappings": {
            "properties": {
                "chunk_id": {"type": "keyword"},
                "source": {"type": "keyword"},
                "chunk_index": {"type": "integer"},
                "text": {
                    "type": "text",
                    "analyzer": "english_analyzer"
                },
                "embedding": {
                    "type": "dense_vector",
                    "dims": embedding_dim,
                    "index": True,
                    "similarity": "cosine"
                }
            }
        }
    }
    
    es.indices.create(index=index_name, body=index_mapping)
    
    # Index documents
    print(f"   🔄 Indexando documentos...")
    for chunk in tqdm(all_chunks, desc="   Indexing"):
        doc = {
            "chunk_id": chunk['chunk_id'],
            "source": chunk['source'],
            "chunk_index": chunk['chunk_index'],
            "text": chunk['text'],
            "embedding": chunk['embedding']
        }
        es.index(index=index_name, id=chunk['chunk_id'], body=doc)
    
    es.indices.refresh(index=index_name)
    count = es.count(index=index_name)['count']
    print(f"   ✅ {count} documentos indexados")
    
    return True


def main():
    args = parse_args()
    print("=" * 70)
    print("INFERÊNCIA RAG (COM ELASTICSEARCH + OMS)")
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
    # PASSO 2: Configurar Elasticsearch
    # ========================================================================
    print("\n" + "=" * 70)
    print("PASSO 2: CONFIGURANDO ELASTICSEARCH")
    print("=" * 70)
    
    es = Elasticsearch(args.es_host)
    if es.ping():
        info = es.info()
        print(f"✅ Conectado ao Elasticsearch!")
        print(f"   Host: {args.es_host}")
        print(f"   Version: {info['version']['number']}")
    else:
        print("❌ Elasticsearch não está respondendo!")
        print("   Execute: sudo systemctl start elasticsearch")
        return
    
    # Load embedding model
    print(f"\n📥 Carregando modelo de embeddings: {args.embedding_model}")
    embedder = SentenceTransformer(args.embedding_model)
    embedding_dim = embedder.get_sentence_embedding_dimension()
    print(f"   Dimensão: {embedding_dim}")
    
    # Setup index
    if not setup_elasticsearch_index(es, embedder, embedding_dim, args.index_name, args.oms_docs_dir):
        return
    
    # Create retriever
    retriever = HybridRetriever(es, args.index_name, embedder, k=5)
    print("✅ Hybrid Retriever criado!")
    
    # ========================================================================
    # PASSO 3: Carregar Modelo LLM
    # ========================================================================
    print("\n" + "=" * 70)
    print("PASSO 3: CARREGANDO MODELO LLaMA-2-13B-Chat + LoRA Amive")
    print("=" * 70)
    
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
    # PASSO 4: Definir Função de Inferência RAG
    # ========================================================================
    print("\n" + "=" * 70)
    print("PASSO 4: CONFIGURANDO FUNÇÃO DE INFERÊNCIA RAG")
    print("=" * 70)
    
    def generate_rag_response(user_input):
        """Gera uma resposta empática usando RAG."""
        
        # Step 1: Retrieve relevant context
        retrieved_docs = retriever.retrieve(user_input)
        
        # Step 2: Format context (top 3, smart truncation)
        context_parts = []
        for i, doc in enumerate(retrieved_docs[:3]):
            text = doc['text']
            if len(text) > 500:
                cut_point = 500
                for end_char in ['. ', '.\n', '? ', '!\n']:
                    last_end = text[:500].rfind(end_char)
                    if last_end > 200:
                        cut_point = last_end + 1
                        break
                text = text[:cut_point].strip()
            context_parts.append(f"[{i+1}]: {text}")
        context = "\n\n".join(context_parts)
        
        # Step 3: Create prompt with centralized template (RAG without persona)
        prompt = build_rag_prompt(
            retrieved_context=context,
            user_message=user_input,
        )
        
        # Step 4: Generate response
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                do_sample=True,
                top_p=args.top_p,
                pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=args.repetition_penalty
            )
        
        # Decode only new tokens
        input_length = inputs['input_ids'].shape[1]
        response_tokens = outputs[0][input_length:]
        response = tokenizer.decode(response_tokens, skip_special_tokens=True).strip()
        
        return response
    
    print("✅ Função de inferência RAG definida!")
    print(f"   Max tokens: {args.max_new_tokens}")
    print(f"   Temperature: {args.temperature}")
    print(f"   Top-p: {args.top_p}")
    print(f"   Repetition penalty: {args.repetition_penalty}")
    
    # ========================================================================
    # PASSO 5: Carregar Dataset
    # ========================================================================
    print("\n" + "=" * 70)
    print("PASSO 5: CARREGANDO DATASET")
    print("=" * 70)
    
    df = pd.read_excel(args.dataset)
    print(f"✅ Dataset carregado: {len(df)} instâncias")
    print(f"   Colunas: {df.columns.tolist()}")
    
    # Verificar se já existe checkpoint
    if os.path.exists(args.checkpoint):
        df_checkpoint = pd.read_excel(args.checkpoint)
        if 'resposta_rag' in df_checkpoint.columns:
            already_done = df_checkpoint['resposta_rag'].notna().sum()
            print(f"\n📂 Checkpoint encontrado: {already_done} instâncias já processadas")
            df = df_checkpoint
        else:
            df['resposta_rag'] = None
            already_done = 0
    else:
        df['resposta_rag'] = None
        already_done = 0
    
    # ========================================================================
    # PASSO 6: Executar Inferências RAG
    # ========================================================================
    print("\n" + "=" * 70)
    print("PASSO 6: EXECUTANDO INFERÊNCIAS RAG")
    print("=" * 70)
    
    pending = df[df['resposta_rag'].isna()]
    print(f"   Pendentes: {len(pending)} instâncias")
    print(f"   Tempo estimado: {len(pending) * 2 / 60:.1f} - {len(pending) * 4 / 60:.1f} horas")
    
    processed = 0
    errors = 0
    
    for idx, row in tqdm(pending.iterrows(), total=len(pending), desc="RAG Inference"):
        try:
            response = generate_rag_response(row['input'])
            df.at[idx, 'resposta_rag'] = response
            processed += 1
            
            # Checkpoint
            if processed % args.checkpoint_every == 0:
                ensure_parent_dir(args.checkpoint)
                df.to_excel(args.checkpoint, index=False)
                tqdm.write(f"   💾 Checkpoint salvo ({processed} processados)")
                
        except Exception as e:
            errors += 1
            df.at[idx, 'resposta_rag'] = f"ERRO: {str(e)}"
            tqdm.write(f"   ❌ Erro no ID {row['id']}: {e}")
    
    # ========================================================================
    # PASSO 7: Salvar Resultado Final
    # ========================================================================
    print("\n" + "=" * 70)
    print("PASSO 7: SALVANDO RESULTADO")
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
    print(f"   Colunas finais: {df.columns.tolist()}")
    
    # Estatísticas
    print("\n📊 ESTATÍSTICAS:")
    df['resposta_rag_len'] = df['resposta_rag'].apply(lambda x: len(str(x).split()) if pd.notna(x) else 0)
    for dataset in df['dataset'].unique():
        subset = df[df['dataset'] == dataset]
        mean_len = subset['resposta_rag_len'].mean()
        print(f"   {dataset}: {mean_len:.0f} palavras (média)")
    
    print(f"\n✅ Finalizado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
