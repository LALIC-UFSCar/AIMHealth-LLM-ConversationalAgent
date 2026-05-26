
import os
from pathlib import Path
import PyPDF2
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from tqdm import tqdm

# --- Configurações ---
PDF_DIRECTORY = Path("/home/rafael/docs/oms")
ELASTICSEARCH_HOST = "http://localhost:9200"
INDEX_NAME = "oms_docs"
CHUNK_SIZE = 500  # Caracteres por pedaço
CHUNK_OVERLAP = 50 # Sobreposição entre pedaços para manter o contexto

# --- Conexão com o Elasticsearch ---
try:
    es = Elasticsearch(ELASTICSEARCH_HOST)
    if not es.ping():
        raise ConnectionError("Não foi possível conectar ao Elasticsearch.")
    print("Conectado ao Elasticsearch com sucesso!")
except ConnectionError as e:
    print(f"Erro de conexão com o Elasticsearch: {e}")
    print("Por favor, verifique se o Elasticsearch está rodando em http://localhost:9200")
    exit()

def create_index_if_not_exists():
    """Cria o índice no Elasticsearch se ele não existir."""
    if not es.indices.exists(index=INDEX_NAME):
        print(f"Criando índice '{INDEX_NAME}'...")
        es.indices.create(
            index=INDEX_NAME,
            body={
                "mappings": {
                    "properties": {
                        "content": {"type": "text", "analyzer": "standard"},
                        "source": {"type": "keyword"},
                    }
                }
            },
        )
    else:
        print(f"Índice '{INDEX_NAME}' já existe.")

def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extrai texto de um arquivo PDF."""
    try:
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            text = "".join(page.extract_text() for page in reader.pages if page.extract_text())
            return text
    except Exception as e:
        print(f"Erro ao ler o PDF {pdf_path}: {e}")
        return ""

def chunk_text(text: str, source: str) -> list[dict]:
    """Divide o texto em pedaços com sobreposição."""
    if not text:
        return []

    chunks = []
    for i in range(0, len(text), CHUNK_SIZE - CHUNK_OVERLAP):
        chunk = text[i:i + CHUNK_SIZE]
        chunks.append({
            "_index": INDEX_NAME,
            "_source": {
                "content": chunk,
                "source": source,
            }
        })
    return chunks

def main():
    """Função principal para indexar os documentos."""
    create_index_if_not_exists()

    pdf_files = list(PDF_DIRECTORY.glob("*.pdf"))
    if not pdf_files:
        print(f"Nenhum arquivo PDF encontrado em {PDF_DIRECTORY}")
        return

    print(f"Encontrados {len(pdf_files)} arquivos PDF para indexar.")

    actions = []
    for pdf_path in tqdm(pdf_files, desc="Processando PDFs"):
        text = extract_text_from_pdf(pdf_path)
        if text:
            chunks = chunk_text(text, pdf_path.name)
            actions.extend(chunks)

    if not actions:
        print("Nenhum texto para indexar.")
        return

    print(f"\nIndexando {len(actions)} pedaços de texto no Elasticsearch...")
    try:
        success, failed = bulk(es, actions)
        print(f"Indexação concluída: {success} sucesso, {failed} falhas.")
    except Exception as e:
        print(f"Ocorreu um erro durante a indexação em massa: {e}")

if __name__ == "__main__":
    main()
