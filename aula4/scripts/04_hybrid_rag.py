"""
04_hybrid_rag.py - RAG com busca hibrida (BM25 + densa + RRF) -> Groq.

Pipeline da Aula 4:
  1. Gera o embedding da pergunta (Ollama).
  2. Busca em paralelo: BM25 (palavras) e densa (kNN/significado).
  3. Funde as duas listas com RRF (DocumentJoiner).
  4. Monta o prompt com os melhores trechos e responde com a Groq, citando fontes.

Observabilidade: se as chaves do LangFuse estiverem no .env, o pipeline e
auto-instrumentado e um link do trace aparece no final.

Antes de rodar, indexe com:  python 01_indexar_hibrido.py

Precisa de OpenSearch, Ollama, Groq (e LangFuse opcional).

Uso:
    python 04_hybrid_rag.py --pergunta "Quais os requisitos para operacao de credito com garantia da Uniao?"
    python 04_hybrid_rag.py --pergunta "..." --top-k 5
"""

import argparse

import _comum

_comum.carregar_env()  # antes de importar haystack (liga o tracing do LangFuse)

from haystack import Pipeline                                                 # noqa: E402
from haystack.components.builders import PromptBuilder                         # noqa: E402
from haystack.components.generators import OpenAIGenerator                     # noqa: E402
from haystack.components.joiners import DocumentJoiner                         # noqa: E402
from haystack.utils import Secret                                             # noqa: E402
from haystack_integrations.components.embedders.ollama import OllamaTextEmbedder  # noqa: E402
from haystack_integrations.components.retrievers.opensearch import (          # noqa: E402
    OpenSearchBM25Retriever,
    OpenSearchEmbeddingRetriever,
)
from haystack_integrations.document_stores.opensearch import OpenSearchDocumentStore  # noqa: E402

TEMPLATE = """
Voce e um assistente juridico especializado em direito e controle externo (TCU).
Responda APENAS com base nos trechos abaixo. Se a informacao nao estiver neles,
diga: "Essa informacao nao consta nos documentos disponiveis." Cite o titulo das
fontes usadas. NAO invente leis nem fatos.

Trechos recuperados:
{% for doc in documents %}
[{{ loop.index }}] {{ doc.meta.titulo }}
{{ doc.content }}
{% endfor %}

Pergunta: {{ question }}

Resposta fundamentada (com as fontes):
"""


def montar_pipeline(indice, top_k, usar_langfuse):
    base_url, modelo_embed = _comum.config_ollama()
    os_cfg = _comum.config_opensearch()
    auth = (os_cfg["usuario"], os_cfg["senha"]) if os_cfg["usuario"] else None
    groq_key, groq_modelo, groq_base = _comum.config_groq()

    store = OpenSearchDocumentStore(
        hosts=os_cfg["url"], index=indice, embedding_dim=_comum.dimensao_do_modelo(modelo_embed),
        http_auth=auth, use_ssl=False, verify_certs=False,
    )

    pipe = Pipeline()
    if usar_langfuse:
        from haystack_integrations.components.connectors.langfuse import LangfuseConnector

        pipe.add_component("tracer", LangfuseConnector("hybrid-rag-aula4"))

    pipe.add_component("embedder", OllamaTextEmbedder(model=modelo_embed, url=base_url))
    pipe.add_component("bm25", OpenSearchBM25Retriever(document_store=store, top_k=top_k))
    pipe.add_component("denso", OpenSearchEmbeddingRetriever(document_store=store, top_k=top_k))
    # DocumentJoiner funde as duas listas usando RRF (Reciprocal Rank Fusion).
    pipe.add_component("juntar", DocumentJoiner(join_mode="reciprocal_rank_fusion", top_k=top_k))
    pipe.add_component("prompt", PromptBuilder(template=TEMPLATE, required_variables=["documents", "question"]))
    pipe.add_component("llm", OpenAIGenerator(
        api_key=Secret.from_token(groq_key), model=groq_modelo, api_base_url=groq_base,
        generation_kwargs={"temperature": 0.2, "max_tokens": 500},
    ))

    pipe.connect("embedder.embedding", "denso.query_embedding")
    pipe.connect("bm25.documents", "juntar.documents")
    pipe.connect("denso.documents", "juntar.documents")
    pipe.connect("juntar.documents", "prompt.documents")
    pipe.connect("prompt.prompt", "llm.prompt")
    return pipe


def main():
    parser = argparse.ArgumentParser(description="RAG hibrido (BM25+densa+RRF) -> Groq.")
    parser.add_argument("--pergunta", required=True, help="a pergunta a responder")
    parser.add_argument("--indice", default="aula4_hibrido", help="indice no OpenSearch")
    parser.add_argument("--top-k", type=int, default=5, help="quantos trechos usar")
    args = parser.parse_args()

    print("=" * 60)
    print("  HYBRID RAG (BM25 + densa + RRF) - Aula 4")
    print("=" * 60)
    print(f"Pergunta: {args.pergunta}")

    usar_langfuse = _comum.langfuse_configurado()
    print(f"LangFuse: {'ligado' if usar_langfuse else 'desligado (sem chaves no .env)'}")

    pipe = montar_pipeline(args.indice, args.top_k, usar_langfuse)
    resultado = pipe.run(
        {
            "embedder": {"text": args.pergunta},
            "bm25": {"query": args.pergunta},
            "prompt": {"question": args.pergunta},
        },
        include_outputs_from={"juntar"},
    )

    print("\nFONTES (apos fusao RRF):")
    print("-" * 50)
    for i, doc in enumerate(resultado["juntar"]["documents"], 1):
        print(f"  [{i}] [{(doc.score or 0):.3f}] {doc.meta.get('titulo', '')[:65]}")

    print("\nRESPOSTA DO LLM:")
    print("-" * 50)
    print(resultado["llm"]["replies"][0])

    if usar_langfuse and "tracer" in resultado and resultado["tracer"].get("trace_url"):
        print(f"\nTrace no LangFuse: {resultado['tracer']['trace_url']}")


if __name__ == "__main__":
    main()
