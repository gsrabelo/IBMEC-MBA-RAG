"""
config.py - Configuracao central do Projeto Final (Aula 12).

Le o .env e expoe as configuracoes de Groq (LLM), Ollama (embeddings), OpenSearch
(indexacao) e LightRAG (grafo). Mantido simples e comentado para o aluno.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

PASTA_APP = Path(__file__).resolve().parent
PASTA_PROJETO = PASTA_APP.parent
PASTA_UPLOADS = PASTA_PROJETO / "uploads"            # arquivos enviados
PASTA_RAG_STORAGE = PASTA_PROJETO / "rag_storage"    # working_dir do LightRAG (grafo)

for _p in (PASTA_UPLOADS, PASTA_RAG_STORAGE):
    _p.mkdir(parents=True, exist_ok=True)

load_dotenv(PASTA_PROJETO / ".env")

# embedding: nomic-embed-text (768) - mesmo modelo no OpenSearch e no LightRAG
DIMENSAO_EMBEDDING = {"nomic-embed-text": 768, "mxbai-embed-large": 1024, "bge-m3": 1024}


def config_groq():
    return (os.getenv("GROQ_API_KEY", ""),
            os.getenv("LLM_MODEL", "llama-3.3-70b-versatile"),
            "https://api.groq.com/openai/v1")


def config_ollama():
    return (os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            os.getenv("EMBEDDING_MODEL", "nomic-embed-text"))


def config_opensearch():
    host = os.getenv("OPENSEARCH_HOST", "localhost")
    porta = os.getenv("OPENSEARCH_PORT", "9200")
    return {"url": f"http://{host}:{porta}",
            "usuario": os.getenv("OPENSEARCH_USER", ""),
            "senha": os.getenv("OPENSEARCH_PASS", ""),
            "indice": os.getenv("OPENSEARCH_INDEX", "projeto_final")}


def dimensao_embedding():
    _, modelo = config_ollama()
    return DIMENSAO_EMBEDDING.get(modelo.split(":")[0].lower(), 768)


def api_keys():
    """Chaves aceitas pela API (separadas por virgula no .env). Vazio = sem auth."""
    bruto = os.getenv("API_KEYS", "").strip()
    return [k.strip() for k in bruto.split(",") if k.strip()]


def langfuse_configurado():
    return bool(os.getenv("LANGFUSE_SECRET_KEY") and os.getenv("LANGFUSE_PUBLIC_KEY"))
