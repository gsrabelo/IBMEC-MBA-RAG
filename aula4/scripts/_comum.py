"""
_comum.py - Funcoes auxiliares compartilhadas pelos scripts da Aula 4.

Evita repeticao: carregar .env, ler o corpus e as queries, configuracoes de
OpenSearch / Ollama / Groq / LangFuse, cliente Groq.

Voce NAO executa este arquivo diretamente. Ele e importado pelos outros scripts.

IMPORTANTE: chame carregar_env() ANTES de importar o haystack nos scripts que
usam LangFuse (a auto-instrumentacao depende de HAYSTACK_CONTENT_TRACING_ENABLED).
"""

import json
import os
from pathlib import Path

from dotenv import load_dotenv

PASTA_SCRIPTS = Path(__file__).resolve().parent           # .../aula4/scripts
PASTA_AULA4 = PASTA_SCRIPTS.parent                        # .../aula4
PASTA_PROJETO = PASTA_AULA4.parent                        # .../MBA_RAG_CAG
PASTA_DATASETS = PASTA_AULA4 / "datasets"
# corpus grande (1100 acordaos do TCU) - alinhado com o gabarito das queries
CORPUS = PASTA_DATASETS / "corpus_juridico_aula4_v2.json"
QUERIES = PASTA_DATASETS / "queries_avaliacao_aula4.json"


def langfuse_configurado():
    return bool(os.getenv("LANGFUSE_SECRET_KEY") and os.getenv("LANGFUSE_PUBLIC_KEY"))


def carregar_env():
    caminho = None
    for c in [PASTA_PROJETO / ".env", PASTA_AULA4 / ".env", PASTA_SCRIPTS / ".env"]:
        if c.exists():
            load_dotenv(c)
            caminho = c
            break
    base = os.getenv("LANGFUSE_BASE_URL")
    if base and not os.getenv("LANGFUSE_HOST"):
        os.environ["LANGFUSE_HOST"] = base
    if langfuse_configurado():
        os.environ["HAYSTACK_CONTENT_TRACING_ENABLED"] = "true"
    return caminho


def carregar_corpus(limite=0):
    """Le o corpus (lista de docs). limite>0 corta nos N primeiros (testes rapidos).

    Cada documento ganha 'texto' = titulo + conteudo.
    """
    with open(CORPUS, "r", encoding="utf-8") as f:
        docs = json.load(f)
    if limite and limite > 0:
        docs = docs[:limite]
    for d in docs:
        d["texto"] = f"{d.get('titulo', '')}. {d.get('conteudo', '')}".strip()
    return docs


def carregar_queries():
    """Le as queries de avaliacao: lista de {id, texto, documentos_relevantes}."""
    with open(QUERIES, "r", encoding="utf-8") as f:
        return json.load(f)


def config_ollama():
    return (os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text"))


def config_opensearch():
    host = os.getenv("OPENSEARCH_HOST", "localhost")
    porta = os.getenv("OPENSEARCH_PORT", "9200")
    usuario = os.getenv("OPENSEARCH_USER", "")
    senha = os.getenv("OPENSEARCH_PASS", "")
    return {"url": f"http://{host}:{porta}", "usuario": usuario, "senha": senha}


def config_groq():
    return (os.getenv("GROQ_API_KEY", ""),
            os.getenv("GROQ_LLM_MODEL", "llama-3.1-8b-instant"),
            "https://api.groq.com/openai/v1")


DIMENSAO_EMBEDDING = {"nomic-embed-text": 768, "mxbai-embed-large": 1024, "bge-m3": 1024}


def dimensao_do_modelo(nome_modelo):
    return DIMENSAO_EMBEDDING.get(nome_modelo.split(":")[0].lower(), 768)


def groq_client():
    """Devolve (cliente OpenAI apontando para a Groq, modelo)."""
    from openai import OpenAI

    api_key, modelo, base_url = config_groq()
    return OpenAI(api_key=api_key, base_url=base_url), modelo
