"""
_comum.py - Funcoes auxiliares compartilhadas pelos scripts da Aula 7.

Aula 7 = Query Enhancement (Multi-Query, Step-Back, RAG-Fusion) com Haystack.
Stack: Haystack + Ollama (embeddings) + Groq (LLM) + OpenSearch (vetores) +
LangFuse (observabilidade no chat).

Corpus/indice: REUSA os acordaos do TCU (mesmo conteudo do indice 'aula4_hibrido').

Voce NAO executa este arquivo diretamente. Ele e importado pelos outros scripts.
"""

import importlib.util
import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv

PASTA_SCRIPTS = Path(__file__).resolve().parent
PASTA_AULA7 = PASTA_SCRIPTS.parent
PASTA_PROJETO = PASTA_AULA7.parent
PASTA_DATASETS = PASTA_AULA7 / "datasets"

INDICE_TCU = os.getenv("AULA7_INDICE", "aula4_hibrido")
CORPUS_ACORDAOS_AULA4 = PASTA_PROJETO / "aula4" / "datasets" / "corpus_juridico_aula4_v2.json"
BENCHMARK = PASTA_DATASETS / "benchmark_gerado.json"


def langfuse_configurado():
    return bool(os.getenv("LANGFUSE_SECRET_KEY") and os.getenv("LANGFUSE_PUBLIC_KEY"))


def carregar_env():
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    caminho = None
    for c in [PASTA_PROJETO / ".env", PASTA_AULA7 / ".env", PASTA_SCRIPTS / ".env"]:
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


def carregar_acordaos_aula4(limite=0):
    """Le os acordaos do TCU (Aula 4). Cada doc ganha 'texto' = titulo + conteudo."""
    with open(CORPUS_ACORDAOS_AULA4, "r", encoding="utf-8") as f:
        docs = json.load(f)
    if limite and limite > 0:
        docs = docs[:limite]
    for d in docs:
        d["texto"] = f"{d.get('titulo', '')}. {d.get('conteudo', '')}".strip()
    return docs


def documentos_haystack(limite=0):
    """Acordaos do TCU como Documents do Haystack (para o 01 indexar)."""
    from haystack import Document

    return [
        Document(content=d["texto"],
                 meta={"id_original": d["id"], "tipo": d.get("tipo", "acordao")})
        for d in carregar_acordaos_aula4(limite=limite)
    ]


def carregar_benchmark():
    if not BENCHMARK.exists():
        raise FileNotFoundError(
            f"{BENCHMARK.name} nao encontrado. Rode primeiro: python 05_benchmark.py (ele gera).")
    with open(BENCHMARK, "r", encoding="utf-8") as f:
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


def abrir_store(indice):
    from haystack_integrations.document_stores.opensearch import OpenSearchDocumentStore

    _, modelo = config_ollama()
    os_cfg = config_opensearch()
    auth = (os_cfg["usuario"], os_cfg["senha"]) if os_cfg["usuario"] else None
    return OpenSearchDocumentStore(
        hosts=os_cfg["url"], index=indice, embedding_dim=dimensao_do_modelo(modelo),
        http_auth=auth, use_ssl=False, verify_certs=False,
    )


def doc_embedder():
    from haystack_integrations.components.embedders.ollama import OllamaDocumentEmbedder

    base_url, modelo = config_ollama()
    return OllamaDocumentEmbedder(model=modelo, url=base_url)


def text_embedder():
    from haystack_integrations.components.embedders.ollama import OllamaTextEmbedder

    base_url, modelo = config_ollama()
    return OllamaTextEmbedder(model=modelo, url=base_url)


def montar_busca(store, top_k):
    """Pipeline simples de busca densa: embedder (Ollama) -> retriever (OpenSearch)."""
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
    """Roda a busca densa para uma query e devolve a lista de Documents."""
    if not query or not query.strip():
        return []
    return pipe.run({"embedder": {"text": query}})["retriever"]["documents"]


def groq_client():
    from openai import OpenAI

    api_key, modelo, base_url = config_groq()
    return OpenAI(api_key=api_key, base_url=base_url), modelo


def _eh_reasoning(modelo):
    """Heuristica: o modelo e de 'raciocinio' (gasta tokens pensando)?"""
    m = (modelo or "").lower()
    return any(t in m for t in ["gpt-oss", "deepseek-r1", "qwq", "o1", "o3", "reason"])


def gerar_texto(cliente, modelo, prompt, max_tokens=400, temperature=0.5):
    """Chama o LLM e devolve o texto, robusto a modelos de RACIOCINIO.

    Modelos de reasoning (ex.: gpt-oss) consomem o orcamento de tokens 'pensando' e,
    com max_tokens baixo, devolvem content VAZIO. Aqui: pedimos reasoning_effort baixo,
    damos folga de tokens, protegemos content None/"" e removemos blocos <think>.
    """
    kwargs = dict(model=modelo, messages=[{"role": "user", "content": prompt}],
                  temperature=temperature, max_tokens=max_tokens)
    if _eh_reasoning(modelo):
        kwargs["max_tokens"] = max(max_tokens, 1024)
        kwargs["extra_body"] = {"reasoning_effort": "low"}
    try:
        resp = cliente.chat.completions.create(**kwargs)
    except Exception:
        kwargs.pop("extra_body", None)  # SDK/endpoint sem suporte a reasoning_effort
        resp = cliente.chat.completions.create(**kwargs)
    conteudo = (resp.choices[0].message.content or "").strip()
    return re.sub(r"<think>.*?</think>", "", conteudo, flags=re.DOTALL).strip()


def gerar_variacoes(cliente, modelo, query, n=4):
    """Gera N variacoes da pergunta (Multi-Query / RAG-Fusion)."""
    prompt = (f"Gere {n} variacoes da pergunta juridica abaixo, com vocabulario diferente "
              f"(sinonimos, termos tecnicos). Uma por linha, sem numeracao.\n\nPergunta: {query}")
    texto = gerar_texto(cliente, modelo, prompt, max_tokens=300, temperature=0.7)
    return [v.strip(" -.") for v in texto.splitlines() if v.strip()][:n]


def gerar_stepback(cliente, modelo, query):
    """Gera uma pergunta mais GERAL (Step-Back Prompting)."""
    prompt = ("Dada a pergunta especifica abaixo, formule UMA pergunta mais GERAL sobre o "
              "conceito juridico por tras dela. Responda apenas com a pergunta geral.\n\n"
              f"Pergunta especifica: {query}")
    return gerar_texto(cliente, modelo, prompt, max_tokens=400, temperature=0.3)


PROMPT_RESPOSTA = (
    "Voce e um assistente juridico especializado em controle externo (TCU). "
    "Responda APENAS com base nos trechos abaixo, de forma objetiva. "
    "Se nao constar, diga que nao consta.\n\nTrechos:\n{contextos}\n\nPergunta: {pergunta}\nResposta:"
)


def responder_com_contexto(cliente, modelo, pergunta, contextos):
    bloco = "\n".join(f"- {c}" for c in contextos) if contextos else "(sem contexto)"
    return gerar_texto(cliente, modelo, PROMPT_RESPOSTA.format(contextos=bloco, pergunta=pergunta),
                       max_tokens=500, temperature=0.2)


def dedup_por_id(listas, top_k):
    """Junta varias listas de Documents, deduplica por id e ORDENA pelo melhor score.

    Ordenar pelo melhor score (em vez da ordem das listas) e essencial: senao a lista
    da pergunta ORIGINAL encheria a cota de top_k e o Multi-Query/Step-Back viraria o
    Baseline. Assim, documentos achados pelas VARIACOES tambem podem entrar no top_k.
    """
    melhor = {}
    for docs in listas:
        for d in docs:
            atual = melhor.get(d.id)
            if atual is None or (d.score or 0.0) > (atual.score or 0.0):
                melhor[d.id] = d
    return sorted(melhor.values(), key=lambda d: (d.score or 0.0), reverse=True)[:top_k]


def fundir_rrf(listas, top_k):
    """Funde varias listas de Documents com Reciprocal Rank Fusion (RAG-Fusion)."""
    from haystack.components.joiners import DocumentJoiner

    joiner = DocumentJoiner(join_mode="reciprocal_rank_fusion", top_k=top_k)
    return joiner.run(documents=listas)["documents"]


def importar_script(nome_arquivo):
    """Importa um script irmao (nome comeca com numero) como modulo."""
    caminho = PASTA_SCRIPTS / nome_arquivo
    spec = importlib.util.spec_from_file_location(nome_arquivo.replace(".py", ""), caminho)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo
