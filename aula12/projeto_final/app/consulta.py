"""
consulta.py - RAG de consulta, roteado para o storage certo.

destino:
  - 'opensearch' (ou 'auto'): busca densa (Ollama + OpenSearch) + geracao (Groq).
  - 'grafo'                  : consulta o LightRAG (modo hibrido) sobre o grafo.

Mantido simples: monta o contexto, gera a resposta e devolve as fontes.
"""

import asyncio

from . import config, indexacao


def _groq():
    from openai import OpenAI

    api_key, modelo, base_url = config.config_groq()
    return OpenAI(api_key=api_key, base_url=base_url), modelo


PROMPT = ("Voce e um assistente juridico. Responda APENAS com base nos trechos abaixo, de "
          "forma objetiva. Se nao constar, diga que nao consta.\n\nTrechos:\n{ctx}\n\n"
          "Pergunta: {q}\nResposta:")


def consultar_opensearch(pergunta, top_k):
    from haystack import Pipeline
    from haystack_integrations.components.embedders.ollama import OllamaTextEmbedder
    from haystack_integrations.components.retrievers.opensearch import (
        OpenSearchEmbeddingRetriever,
    )

    base_url, modelo = config.config_ollama()
    store = indexacao._store_opensearch()
    pipe = Pipeline()
    pipe.add_component("embedder", OllamaTextEmbedder(model=modelo, url=base_url))
    pipe.add_component("retriever", OpenSearchEmbeddingRetriever(document_store=store, top_k=top_k))
    pipe.connect("embedder.embedding", "retriever.query_embedding")
    docs = pipe.run({"embedder": {"text": pergunta}})["retriever"]["documents"]

    cliente, gmodelo = _groq()
    ctx = "\n".join(f"- {d.content}" for d in docs) or "(sem contexto)"
    resp = cliente.chat.completions.create(
        model=gmodelo, messages=[{"role": "user", "content": PROMPT.format(ctx=ctx, q=pergunta)}],
        temperature=0.2, max_tokens=500)
    fontes = [{"id": d.meta.get("id_original") or d.meta.get("arquivo"),
               "trecho": d.content[:160]} for d in docs]
    return (resp.choices[0].message.content or "").strip(), fontes


def consultar_grafo(pergunta):
    async def _run():
        from lightrag import QueryParam
        rag = await indexacao._criar_lightrag()
        try:
            return await rag.aquery(pergunta, param=QueryParam(mode="hybrid"))
        finally:
            await rag.finalize_storages()
    resposta = asyncio.run(_run())
    return resposta, [{"id": "grafo", "trecho": "(resposta sintetizada do grafo de conhecimento)"}]


def consultar(pergunta, destino="auto", top_k=5):
    if destino == "grafo":
        resp, fontes = consultar_grafo(pergunta)
        return resp, fontes, "grafo"
    resp, fontes = consultar_opensearch(pergunta, top_k)
    return resp, fontes, "opensearch"
