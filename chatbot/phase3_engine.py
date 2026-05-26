"""Reusable LangChain + Groq RAG engine for the Streamlit chatbot."""

from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

from config import DEFAULT_OMS_DOCS_DIR, init_env, path_from_env
from inference.guardrail import build_crisis_response_988, detect_crisis_risk
from inference.prompt_templates import build_phase3_prompt


DEFAULT_GROQ_MODEL = "llama-3.1-8b-instant"

MENTAL_HEALTH_KEYWORDS = [
    "anxious", "anxiety", "panic", "worried", "stress", "stressed", "overwhelmed",
    "depressed", "depression", "sad", "sadness", "hopeless", "hopelessness",
    "lonely", "loneliness", "isolated", "alone", "empty", "numb",
    "scared", "fear", "afraid", "terrified", "nervous",
    "angry", "frustrated", "irritable", "upset",
    "exhausted", "tired", "burnout", "drained",
    "worthless", "guilty", "shame", "self-esteem",
    "crying", "cant sleep", "insomnia", "nightmares",
    "panic attack", "heart racing", "breathing", "chest",
    "struggling", "suffering", "hurting", "pain",
    "help", "support", "therapy", "therapist", "counseling",
    "feeling", "emotions", "emotional", "mental health",
    "relationship", "breakup", "divorce", "grief", "loss", "death",
    "job", "work", "school", "exam", "deadline", "pressure",
    "family", "friend", "conflict",
]


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {raw!r}") from exc


@dataclass(frozen=True)
class Phase3Config:
    """Runtime configuration for the LangChain + Groq RAG chatbot."""

    groq_model: str = DEFAULT_GROQ_MODEL
    groq_api_key: str = ""
    oms_docs_dir: Path = DEFAULT_OMS_DOCS_DIR
    es_host: str = "http://localhost:9200"
    es_index: str = "oms_mental_health_docs"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    top_k: int = 5
    candidate_pool_size: int = 50
    rrf_k: int = 60
    max_tokens: int = 600
    temperature: float = 0.7
    max_history_messages: int = 6

    @classmethod
    def from_env(cls) -> "Phase3Config":
        init_env()
        return cls(
            groq_model=os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL),
            groq_api_key=os.getenv("GROQ_API_KEY", ""),
            oms_docs_dir=path_from_env("PERSONARAG_OMS_DOCS_DIR", DEFAULT_OMS_DOCS_DIR),
            es_host=os.getenv("PERSONARAG_ES_HOST", "http://localhost:9200"),
            es_index=os.getenv("PERSONARAG_ES_INDEX", "oms_mental_health_docs"),
            embedding_model=os.getenv(
                "PERSONARAG_EMBEDDING_MODEL",
                "sentence-transformers/all-MiniLM-L6-v2",
            ),
            top_k=_env_int("PERSONARAG_TOP_K", 5),
            candidate_pool_size=_env_int("PERSONARAG_CANDIDATE_POOL_SIZE", 50),
            rrf_k=_env_int("PERSONARAG_RRF_K", 60),
            max_tokens=_env_int("GROQ_MAX_TOKENS", 600),
            temperature=_env_float("GROQ_TEMPERATURE", 0.7),
            max_history_messages=_env_int("PERSONARAG_MAX_HISTORY_MESSAGES", 6),
        )


@dataclass
class Phase3Result:
    response: str
    retrieved_docs: List[Dict[str, Any]] = field(default_factory=list)
    debug: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class HybridRetriever:
    """Hybrid retriever over the Phase 3 Elasticsearch schema."""

    def __init__(
        self,
        es_client: Elasticsearch,
        index_name: str,
        embedder: SentenceTransformer,
        top_k: int = 5,
        candidate_pool_size: int = 50,
        rrf_k: int = 60,
    ):
        self.es = es_client
        self.index_name = index_name
        self.embedder = embedder
        self.top_k = top_k
        self.candidate_pool_size = candidate_pool_size
        self.rrf_k = rrf_k

    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        """Retrieve documents using BM25 + dense search fused with RRF."""
        try:
            query_embedding = self.embedder.encode(query).tolist()

            bm25_body = {
                "size": self.candidate_pool_size,
                "query": {
                    "match": {
                        "text": {
                            "query": query,
                        }
                    }
                },
            }

            dense_body = {
                "size": self.candidate_pool_size,
                "query": {
                    "script_score": {
                        "query": {"match_all": {}},
                        "script": {
                            "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                            "params": {"query_vector": query_embedding},
                        },
                    }
                },
            }

            bm25_response = self.es.search(index=self.index_name, body=bm25_body)
            dense_response = self.es.search(index=self.index_name, body=dense_body)

            return self._fuse_with_rrf(
                bm25_response["hits"]["hits"],
                dense_response["hits"]["hits"],
            )
        except Exception as exc:
            print(f"   RRF retrieval error: {exc}")
            return []

    def _fuse_with_rrf(self, bm25_hits: List[Dict[str, Any]], dense_hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Fuse BM25 and dense-vector rankings using Reciprocal Rank Fusion."""
        fused: Dict[str, Dict[str, Any]] = {}

        def add_hits(hits: List[Dict[str, Any]], source_name: str) -> None:
            for rank, hit in enumerate(hits, start=1):
                chunk_id = hit["_id"]

                if chunk_id not in fused:
                    fused[chunk_id] = {
                        "hit": hit,
                        "rrf_score": 0.0,
                        "bm25_rank": None,
                        "dense_rank": None,
                        "bm25_score": None,
                        "dense_score": None,
                    }

                fused[chunk_id]["rrf_score"] += 1.0 / (self.rrf_k + rank)

                if source_name == "bm25":
                    fused[chunk_id]["bm25_rank"] = rank
                    fused[chunk_id]["bm25_score"] = hit.get("_score")
                elif source_name == "dense":
                    fused[chunk_id]["dense_rank"] = rank
                    fused[chunk_id]["dense_score"] = hit.get("_score")

        add_hits(bm25_hits, "bm25")
        add_hits(dense_hits, "dense")

        ranked = sorted(
            fused.items(),
            key=lambda item: (-item[1]["rrf_score"], item[0]),
        )

        results: List[Dict[str, Any]] = []
        for chunk_id, item in ranked[: self.top_k]:
            hit = item["hit"]
            source = hit["_source"]
            results.append(
                {
                    "text": source.get("text", ""),
                    "score": item["rrf_score"],
                    "rrf_score": item["rrf_score"],
                    "chunk_id": chunk_id,
                    "source": source.get("source", "unknown"),
                    "chunk_index": source.get("chunk_index"),
                    "bm25_rank": item["bm25_rank"],
                    "dense_rank": item["dense_rank"],
                    "bm25_score": item["bm25_score"],
                    "dense_score": item["dense_score"],
                    "retrieval_method": "rrf",
                }
            )

        return results


def extract_client_text(raw_input: str) -> str:
    if not raw_input or not isinstance(raw_input, str):
        return raw_input or ""

    bracket_matches = re.findall(
        r"\[Client\]\s*(.*?)(?=\[Therapist\]|\[Client\]|$)",
        raw_input,
        re.IGNORECASE | re.DOTALL,
    )
    colon_matches = re.findall(
        r"(?:^|(?<=\n))Client:\s*(.*?)(?=(?:\n)?Therapist:|(?:\n)?Client:|$)",
        raw_input,
        re.IGNORECASE | re.DOTALL,
    )
    all_matches = bracket_matches + colon_matches
    if all_matches:
        client_text = " ".join(m.strip() for m in all_matches if m.strip())
        return client_text if client_text else raw_input
    return raw_input


def is_mental_health_related(text: str) -> bool:
    if not text:
        return False

    text_lower = text.lower()
    keyword_count = sum(1 for kw in MENTAL_HEALTH_KEYWORDS if kw in text_lower)
    first_person_patterns = [
        r"i('m| am| feel| have been| can't| don't)",
        r"my (anxiety|depression|stress|fear|worry)",
        r"i've been (feeling|struggling|dealing)",
    ]
    has_personal = any(re.search(pattern, text_lower) for pattern in first_person_patterns)
    return keyword_count >= 2 or (keyword_count >= 1 and has_personal)


def clean_generated_response(response: str) -> str:
    """Remove wrapper tokens and meta preambles before returning the final reply."""
    if not response:
        return ""

    clean = response.strip()
    if "[/INST]" in clean:
        clean = clean.split("[/INST]")[-1].strip()
    clean = re.sub(r"^<s>\s*", "", clean).strip()
    clean = re.sub(r"^(?:assistant|amive)\s*:\s*", "", clean, flags=re.IGNORECASE).strip()

    meta_markers = [
        "here's a", "here is a", "here's the", "here is the",
        "sample emotional support response", "emotional support response",
        "based on the information", "based on the context", "based on the given prompts",
        "based on the provided", "based on the reference",
    ]
    first_chunk = clean[:500].lower()
    has_meta_preamble = any(marker in first_chunk for marker in meta_markers)

    if has_meta_preamble:
        quoted_match = re.search(r'["“](.*?)["”]', clean, flags=re.DOTALL)
        if quoted_match:
            clean = quoted_match.group(1).strip()
        else:
            preamble_patterns = [
                r"^(?:sure,?\s*)?here(?:'s| is)\s+(?:a|the|one)?\s*(?:sample\s*)?(?:emotional\s+support\s+)?(?:response|reply|answer)\b[\s\S]{0,900}?:\s*",
                r"^based on (?:the )?(?:information|context|reference information|provided information|given prompts)[\s\S]{0,900}?(?:response|reply|answer)[\s\S]{0,900}?:\s*",
                r"^based on (?:the )?(?:information|context|reference information|provided information|given prompts)[\s\S]{0,900}?:\s*",
            ]
            for pattern in preamble_patterns:
                updated = re.sub(pattern, "", clean, count=1, flags=re.IGNORECASE).strip()
                if updated != clean:
                    clean = updated
                    break

    trailing_meta_patterns = [
        r"\n\s*\n\s*this response\b[\s\S]*$",
        r"\n\s*\n\s*the response\b[\s\S]*$",
        r"\n\s*\n\s*it (?:also )?acknowledges\b[\s\S]*$",
        r"\n\s*\n\s*additionally,?\b[\s\S]*$",
    ]
    for pattern in trailing_meta_patterns:
        clean = re.sub(pattern, "", clean, count=1, flags=re.IGNORECASE).strip()

    if len(clean) >= 2 and clean[0] == clean[-1] and clean[0] in {"'", '"'}:
        clean = clean[1:-1].strip()

    return re.sub(r"^(?:assistant|amive)\s*:\s*", "", clean, flags=re.IGNORECASE).strip()


def extract_text_from_pdf(pdf_path: Path) -> Dict[str, Any]:
    reader = PdfReader(str(pdf_path))
    text_parts: List[str] = []
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text_parts.append(page_text)
    text = "\n".join(text_parts)
    return {
        "filename": pdf_path.name,
        "filepath": str(pdf_path),
        "num_pages": len(reader.pages),
        "text": text,
        "char_count": len(text),
    }


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    chunks: List[str] = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = start + chunk_size
        if end < text_len:
            for sep in [". ", ".\n", "\n\n"]:
                last_sep = text[start:end].rfind(sep)
                if last_sep != -1 and last_sep > chunk_size // 2:
                    end = start + last_sep + len(sep)
                    break
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap
    return chunks


class Phase3Engine:
    """Interactive RAG runtime backed by LangChain and Groq."""

    def __init__(self, config: Optional[Phase3Config] = None):
        self.config = config or Phase3Config.from_env()
        self.es: Optional[Elasticsearch] = None
        self.embedder: Optional[SentenceTransformer] = None
        self.retriever: Optional[HybridRetriever] = None
        self.llm: Any = None

    def load(self) -> "Phase3Engine":
        self.load_llm()
        self.connect_elasticsearch()
        self.load_embedder()
        self.ensure_knowledge_base()
        self.create_retriever()
        return self

    def connect_elasticsearch(self) -> Elasticsearch:
        self.es = Elasticsearch(self.config.es_host)
        try:
            available = self.es.ping()
        except Exception as exc:
            raise RuntimeError(f"Elasticsearch is not available at {self.config.es_host}: {exc}") from exc
        if not available:
            raise RuntimeError(f"Elasticsearch is not available at {self.config.es_host}")
        return self.es

    def load_embedder(self) -> SentenceTransformer:
        if self.embedder is None:
            self.embedder = SentenceTransformer(self.config.embedding_model)
        return self.embedder

    def load_llm(self) -> Any:
        if self.llm is not None:
            return self.llm
        if not self.config.groq_api_key:
            raise RuntimeError("GROQ_API_KEY is required for Groq inference. Set it in .env or the environment.")

        try:
            from langchain_groq import ChatGroq
        except ImportError as exc:
            raise RuntimeError(
                "langchain-groq is required for Groq inference. Install dependencies with "
                "`pip install -r requirements.txt`."
            ) from exc

        os.environ["GROQ_API_KEY"] = self.config.groq_api_key
        self.llm = ChatGroq(
            model=self.config.groq_model,
            groq_api_key=self.config.groq_api_key,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        return self.llm

    def ensure_knowledge_base(self, force_rebuild: bool = False) -> int:
        if self.es is None:
            self.connect_elasticsearch()
        if self.embedder is None:
            self.load_embedder()
        assert self.es is not None
        assert self.embedder is not None

        index_name = self.config.es_index
        if self.es.indices.exists(index=index_name):
            count = self.es.count(index=index_name)["count"]
            if count > 0 and not force_rebuild:
                return count
            self.es.indices.delete(index=index_name)

        pdf_dir = Path(self.config.oms_docs_dir).expanduser().resolve()
        if not pdf_dir.exists():
            raise FileNotFoundError(f"WHO/OMS PDF directory not found: {pdf_dir}")

        pdf_files = sorted(pdf_dir.glob("*.pdf"))
        if not pdf_files:
            raise FileNotFoundError(f"No PDF files found in {pdf_dir}")

        documents = [extract_text_from_pdf(pdf_file) for pdf_file in pdf_files]
        chunks: List[Dict[str, Any]] = []
        for document in documents:
            for idx, chunk in enumerate(chunk_text(document["text"])):
                chunks.append(
                    {
                        "chunk_id": f"{document['filename']}_{idx}",
                        "source": document["filename"],
                        "chunk_index": idx,
                        "text": chunk,
                        "char_count": len(chunk),
                    }
                )

        if not chunks:
            raise RuntimeError(f"No text chunks could be created from PDFs in {pdf_dir}")

        texts = [chunk["text"] for chunk in chunks]
        embeddings = self.embedder.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        for idx, chunk in enumerate(chunks):
            chunk["embedding"] = embeddings[idx].tolist()

        mapping = {
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 0,
                "analysis": {"analyzer": {"english_analyzer": {"type": "english"}}},
            },
            "mappings": {
                "properties": {
                    "chunk_id": {"type": "keyword"},
                    "source": {"type": "keyword"},
                    "chunk_index": {"type": "integer"},
                    "text": {"type": "text", "analyzer": "english_analyzer"},
                    "char_count": {"type": "integer"},
                    "embedding": {
                        "type": "dense_vector",
                        "dims": self.embedder.get_sentence_embedding_dimension(),
                        "index": True,
                        "similarity": "cosine",
                    },
                }
            },
        }
        self.es.indices.create(index=index_name, body=mapping)
        actions = [
            {"_index": index_name, "_id": chunk["chunk_id"], "_source": chunk}
            for chunk in chunks
        ]
        bulk(self.es, actions)
        self.es.indices.refresh(index=index_name)
        return self.es.count(index=index_name)["count"]

    def rebuild_knowledge_base(self) -> int:
        count = self.ensure_knowledge_base(force_rebuild=True)
        self.create_retriever()
        return count

    def create_retriever(self) -> HybridRetriever:
        if self.es is None:
            self.connect_elasticsearch()
        if self.embedder is None:
            self.load_embedder()
        assert self.es is not None
        assert self.embedder is not None
        self.retriever = HybridRetriever(
            es_client=self.es,
            index_name=self.config.es_index,
            embedder=self.embedder,
            top_k=self.config.top_k,
            candidate_pool_size=self.config.candidate_pool_size,
            rrf_k=self.config.rrf_k,
        )
        return self.retriever

    def knowledge_base_count(self) -> int:
        if self.es is None:
            self.connect_elasticsearch()
        assert self.es is not None
        if not self.es.indices.exists(index=self.config.es_index):
            return 0
        return self.es.count(index=self.config.es_index)["count"]

    def format_history(self, history: Optional[Iterable[Dict[str, str]]]) -> str:
        if not history:
            return ""
        lines: List[str] = []
        recent = list(history)[-self.config.max_history_messages:]
        for item in recent:
            role = item.get("role", "")
            content = str(item.get("content", "")).strip()
            if not content:
                continue
            if role == "user":
                lines.append(f"User: {content}")
            elif role == "assistant":
                lines.append(f"Assistant: {content}")
        return "\n".join(lines)

    def build_prompt_user_message(self, clean_query: str, history: Optional[Iterable[Dict[str, str]]]) -> str:
        return clean_query

    def _format_retrieved_context(self, retrieved_docs: List[Dict[str, Any]]) -> str:
        context_parts: List[str] = []
        for idx, doc in enumerate(retrieved_docs[:3]):
            text = self._trim_passage(str(doc.get("text", "")), max_chars=500)
            context_parts.append(f"[{idx + 1}]: {text}")
        return "\n\n".join(context_parts)

    @staticmethod
    def _trim_passage(text: str, max_chars: int = 500) -> str:
        if len(text) <= max_chars:
            return text.strip()

        cut_point = max_chars
        for end_char in [". ", ".\n", "? ", "?\n", "! ", "!\n"]:
            last_end = text[:max_chars].rfind(end_char)
            if last_end > 200:
                cut_point = last_end + 1
                break
        return text[:cut_point].strip()

    @staticmethod
    def _message_content_to_text(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                if isinstance(item, dict):
                    parts.append(str(item.get("text") or item.get("content") or ""))
                else:
                    parts.append(str(item))
            return "".join(parts)
        return str(content)

    def generate(
        self,
        user_message: str,
        history: Optional[Iterable[Dict[str, str]]] = None,
    ) -> Phase3Result:
        history_list = list(history) if history else []
        clean_query = extract_client_text(user_message).strip()
        debug: Dict[str, Any] = {
            "clean_query": clean_query[:500],
            "is_mental_health": None,
            "is_crisis": False,
            "crisis_risk_level": "NO_RISK",
            "crisis_signals": [],
            "crisis_path": "normal",
            "history_turns": len(history_list),
            "retrieval_query": clean_query[:500],
            "retrieved_context_chars": 0,
            "es_host": self.config.es_host,
            "es_index": self.config.es_index,
            "groq_model": self.config.groq_model,
            "retrieval_method": "rrf",
            "top_k": self.config.top_k,
            "candidate_pool_size": self.config.candidate_pool_size,
            "rrf_k": self.config.rrf_k,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
        }

        if not clean_query:
            return Phase3Result(
                response="I'm here with you. What would you like to talk about?",
                retrieved_docs=[],
                debug=debug,
            )

        crisis = detect_crisis_risk(clean_query)
        debug["is_crisis"] = bool(crisis["is_crisis"])
        debug["crisis_risk_level"] = str(crisis["risk_level"])
        debug["crisis_signals"] = list(crisis["signals"])
        if crisis["is_crisis"]:
            debug["crisis_path"] = "early_return_988"
            return Phase3Result(response=build_crisis_response_988(), retrieved_docs=[], debug=debug)

        if self.retriever is None:
            self.create_retriever()
        if self.llm is None:
            self.load_llm()
        assert self.retriever is not None
        assert self.llm is not None

        debug["is_mental_health"] = is_mental_health_related(clean_query)
        retrieved_docs = self.retriever.retrieve(clean_query)
        retrieved_context = self._format_retrieved_context(retrieved_docs)
        debug["retrieved_context_chars"] = len(retrieved_context)

        prompt = build_phase3_prompt(
            persona_context="",
            retrieved_context=retrieved_context,
            user_message=self.build_prompt_user_message(clean_query, history_list),
        )

        try:
            from langchain_core.messages import HumanMessage
        except ImportError as exc:
            raise RuntimeError(
                "langchain-core is required for Groq inference. Install dependencies with "
                "`pip install -r requirements.txt`."
            ) from exc

        message = self.llm.invoke([HumanMessage(content=prompt)])
        response = clean_generated_response(self._message_content_to_text(message.content))
        return Phase3Result(response=response, retrieved_docs=retrieved_docs, debug=debug)
