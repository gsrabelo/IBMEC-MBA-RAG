"""
_comum.py - Funcoes auxiliares compartilhadas pelos scripts da Aula 11.

Aula 11 = Tecnicas Complementares (Time-Aware, Compressao/LLMLingua, ColBERT,
Multimodal/CLIP, DSPy). Cada tecnica usa a ferramenta de referencia da sua area;
o LLM continua a Groq e os embeddings o Ollama. Busca densa no OpenSearch.

Corpus: aula11/datasets/corpus_juridico_benchmark.json (30 documentos com data/
vigente/tipo + 20 queries de benchmark).

Voce NAO executa este arquivo diretamente. Ele e importado pelos outros scripts.
"""

import json
import os
from pathlib import Path

from dotenv import load_dotenv

PASTA_SCRIPTS = Path(__file__).resolve().parent
PASTA_AULA11 = PASTA_SCRIPTS.parent
PASTA_PROJETO = PASTA_AULA11.parent
PASTA_DATASETS = PASTA_AULA11 / "datasets"

CORPUS = PASTA_DATASETS / "corpus_juridico_benchmark.json"
INDICE = os.getenv("AULA11_INDICE", "aula11_benchmark")

DIMENSAO_EMBEDDING = {"nomic-embed-text": 768, "mxbai-embed-large": 1024, "bge-m3": 1024}


def langfuse_configurado():
    return bool(os.getenv("LANGFUSE_SECRET_KEY") and os.getenv("LANGFUSE_PUBLIC_KEY"))


def carregar_env():
    for c in [PASTA_PROJETO / ".env", PASTA_AULA11 / ".env", PASTA_SCRIPTS / ".env"]:
        if c.exists():
            load_dotenv(c)
            return c
    return None


def config_groq():
    return (os.getenv("GROQ_API_KEY", ""),
            os.getenv("AULA11_LLM_MODEL", "llama-3.3-70b-versatile"),
            "https://api.groq.com/openai/v1")


def config_ollama():
    return (os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text"))


def config_opensearch():
    host = os.getenv("OPENSEARCH_HOST", "localhost")
    porta = os.getenv("OPENSEARCH_PORT", "9200")
    return {"url": f"http://{host}:{porta}",
            "usuario": os.getenv("OPENSEARCH_USER", ""), "senha": os.getenv("OPENSEARCH_PASS", "")}


def dimensao_do_modelo(nome_modelo):
    return DIMENSAO_EMBEDDING.get(nome_modelo.split(":")[0].lower(), 768)


# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------
def carregar_corpus():
    with open(CORPUS, "r", encoding="utf-8") as f:
        d = json.load(f)
    return d["documentos"]


def carregar_queries():
    with open(CORPUS, "r", encoding="utf-8") as f:
        d = json.load(f)
    return d.get("queries_benchmark", [])


def documentos_haystack():
    from haystack import Document

    docs = []
    for d in carregar_corpus():
        docs.append(Document(content=d["texto"], meta={
            "id_original": d["id"], "fonte": d.get("fonte", ""), "data": d.get("data", ""),
            "vigente": str(d.get("vigente", "True")) == "True", "tipo": d.get("tipo", ""),
        }))
    return docs


# ---------------------------------------------------------------------------
# OpenSearch / embeddings (busca densa - base de varias tecnicas)
# ---------------------------------------------------------------------------
def abrir_store(indice=None):
    from haystack_integrations.document_stores.opensearch import OpenSearchDocumentStore

    _, modelo = config_ollama()
    os_cfg = config_opensearch()
    auth = (os_cfg["usuario"], os_cfg["senha"]) if os_cfg["usuario"] else None
    return OpenSearchDocumentStore(hosts=os_cfg["url"], index=indice or INDICE,
                                   embedding_dim=dimensao_do_modelo(modelo),
                                   http_auth=auth, use_ssl=False, verify_certs=False)


def doc_embedder():
    from haystack_integrations.components.embedders.ollama import OllamaDocumentEmbedder

    base_url, modelo = config_ollama()
    return OllamaDocumentEmbedder(model=modelo, url=base_url)


def text_embedder():
    from haystack_integrations.components.embedders.ollama import OllamaTextEmbedder

    base_url, modelo = config_ollama()
    return OllamaTextEmbedder(model=modelo, url=base_url)


def montar_busca(store, top_k=5):
    from haystack import Pipeline
    from haystack_integrations.components.retrievers.opensearch import (
        OpenSearchEmbeddingRetriever,
    )

    pipe = Pipeline()
    pipe.add_component("embedder", text_embedder())
    pipe.add_component("retriever", OpenSearchEmbeddingRetriever(document_store=store, top_k=top_k))
    pipe.connect("embedder.embedding", "retriever.query_embedding")
    return pipe


def buscar(pipe, query):
    return pipe.run({"embedder": {"text": query}})["retriever"]["documents"]


# ---------------------------------------------------------------------------
# Groq (LLM)
# ---------------------------------------------------------------------------
def groq_client():
    from openai import OpenAI

    api_key, modelo, base_url = config_groq()
    return OpenAI(api_key=api_key, base_url=base_url), modelo


def gerar_texto(cliente, modelo, prompt, max_tokens=500, temperature=0.2):
    resp = cliente.chat.completions.create(
        model=modelo, messages=[{"role": "user", "content": prompt}],
        temperature=temperature, max_tokens=max_tokens)
    return (resp.choices[0].message.content or "").strip()


PROMPT_RESPOSTA = (
    "Voce e um assistente juridico. Responda APENAS com base nos trechos abaixo, de forma "
    "objetiva. Se nao constar, diga que nao consta.\n\nTrechos:\n{contextos}\n\n"
    "Pergunta: {pergunta}\nResposta:")


def responder_com_contexto(cliente, modelo, pergunta, contextos):
    bloco = "\n".join(f"- {c}" for c in contextos) if contextos else "(sem contexto)"
    return gerar_texto(cliente, modelo, PROMPT_RESPOSTA.format(contextos=bloco, pergunta=pergunta))
