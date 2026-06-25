"""
04_advanced_rag.py - Advanced RAG completo (Haystack + OpenSearch + BGE-Reranker + Groq).

Junta as tecnicas da Aula 3 em um pipeline so:
  1. (opcional) Reescreve a pergunta (paraphrase / HyDE / step-back) com a Groq.
  2. Gera o embedding da query (Ollama).
  3. Busca os top-N documentos no OpenSearch (kNN).
  4. Reordena com o BGE-Reranker e mantem os melhores (rerank-k).
  5. Monta o prompt e gera a resposta com a Groq, citando as fontes.

Observabilidade: se as chaves do LangFuse estiverem no .env, o pipeline e
auto-instrumentado e um link do trace e exibido no final.

Antes de rodar, indexe o corpus com:  python 02_indexar_opensearch.py

Uso:
    python 04_advanced_rag.py --pergunta "Podem prender alguem sem mandado?"
    python 04_advanced_rag.py --pergunta "..." --rewrite hyde --top-n 10 --rerank-k 4
"""

import argparse

import _comum

_comum.carregar_env()  # antes de importar haystack (liga o tracing do LangFuse)

from haystack import Pipeline                                                 # noqa: E402
from haystack.components.builders import PromptBuilder                         # noqa: E402
from haystack.components.generators import OpenAIGenerator                     # noqa: E402
from haystack.components.rankers import TransformersSimilarityRanker          # noqa: E402
from haystack.utils import Secret                                             # noqa: E402
from haystack_integrations.components.embedders.ollama import OllamaTextEmbedder  # noqa: E402
from haystack_integrations.components.retrievers.opensearch import (          # noqa: E402
    OpenSearchEmbeddingRetriever,
)
from haystack_integrations.document_stores.opensearch import OpenSearchDocumentStore  # noqa: E402

MODELO_RERANKER = "BAAI/bge-reranker-v2-m3"

TEMPLATE = """
Voce e um assistente juridico especializado em direito e seguranca publica.
Responda APENAS com base nos trechos abaixo. Se a informacao nao estiver neles,
diga: "Essa informacao nao consta nos documentos disponiveis." Cite as fontes
(tipo/tribunal). NAO invente leis nem fatos.

Trechos recuperados:
{% for doc in documents %}
[{{ loop.index }}] ({{ doc.meta.tipo }} - {{ doc.meta.tribunal }})
{{ doc.content }}
{% endfor %}

Pergunta: {{ question }}

Resposta fundamentada (com as fontes):
"""


def montar_pipeline(indice, top_n, rerank_k, usar_langfuse):
    base_url, modelo_embed = _comum.config_ollama()
    os_cfg = _comum.config_opensearch()
    auth = (os_cfg["usuario"], os_cfg["senha"]) if os_cfg["usuario"] else None
    groq_key, groq_modelo, groq_base = _comum.config_groq()

    store = OpenSearchDocumentStore(
        hosts=os_cfg["url"], index=indice, embedding_dim=_comum.dimensao_do_modelo(modelo_embed),
        http_auth=auth, use_ssl=False, verify_certs=False,
    )

    pipe = Pipeline()
    # Conector do LangFuse: instrumenta automaticamente todo o pipeline.
    if usar_langfuse:
        from haystack_integrations.components.connectors.langfuse import LangfuseConnector

        pipe.add_component("tracer", LangfuseConnector("advanced-rag-aula3"))

    pipe.add_component("embedder", OllamaTextEmbedder(model=modelo_embed, url=base_url))
    pipe.add_component("retriever", OpenSearchEmbeddingRetriever(document_store=store, top_k=top_n))
    # Nota: usamos o TransformersSimilarityRanker (estavel com a sentence-transformers
    # do venv). O aviso de "legacy" e inofensivo; o SentenceTransformers... exige ST>=4.
    pipe.add_component("ranker", TransformersSimilarityRanker(model=MODELO_RERANKER, top_k=rerank_k))
    pipe.add_component("prompt", PromptBuilder(template=TEMPLATE, required_variables=["documents", "question"]))
    pipe.add_component("llm", OpenAIGenerator(
        api_key=Secret.from_token(groq_key), model=groq_modelo, api_base_url=groq_base,
        generation_kwargs={"temperature": 0.2, "max_tokens": 500},
    ))

    pipe.connect("embedder.embedding", "retriever.query_embedding")
    pipe.connect("retriever.documents", "ranker.documents")
    pipe.connect("ranker.documents", "prompt.documents")
    pipe.connect("prompt.prompt", "llm.prompt")
    return pipe


def main():
    parser = argparse.ArgumentParser(description="Advanced RAG (rewrite + rerank + LangFuse).")
    parser.add_argument("--pergunta", required=True, help="a pergunta a responder")
    parser.add_argument("--rewrite", default="none",
                        choices=["none", "paraphrase", "hyde", "stepback"],
                        help="reescreve a pergunta antes de buscar (padrao: none)")
    parser.add_argument("--indice", default="aula3_juridico", help="indice no OpenSearch")
    parser.add_argument("--top-n", type=int, default=10, help="quantos buscar (bi-encoder)")
    parser.add_argument("--rerank-k", type=int, default=4, help="quantos manter apos o rerank")
    args = parser.parse_args()

    print("=" * 60)
    print("  ADVANCED RAG - Aula 3")
    print("=" * 60)
    print(f"Pergunta: {args.pergunta}")

    # Passo 1 (opcional): reescrever a query usada na BUSCA.
    query_busca = args.pergunta
    if args.rewrite != "none":
        cliente, modelo = _comum.groq_client()
        query_busca = _comum.reescrever_query(cliente, modelo, args.pergunta, args.rewrite)
        print(f"Query reescrita ({args.rewrite}): {query_busca}")

    usar_langfuse = _comum.langfuse_configurado()
    print(f"LangFuse: {'ligado' if usar_langfuse else 'desligado (sem chaves no .env)'}")
    print("Carregando reranker (1a vez baixa ~560 MB)...")

    pipe = montar_pipeline(args.indice, args.top_n, args.rerank_k, usar_langfuse)
    resultado = pipe.run(
        {
            "embedder": {"text": query_busca},
            "ranker": {"query": query_busca},      # o reranker precisa da query para pontuar
            "prompt": {"question": args.pergunta},
        },
        include_outputs_from={"ranker"},
    )

    print("\nFONTES (apos reranking):")
    print("-" * 50)
    for i, doc in enumerate(resultado["ranker"]["documents"], 1):
        print(f"  [{i}] [{doc.score:.3f}] {doc.meta.get('tipo','')} - {doc.meta.get('tribunal','')}")

    print("\nRESPOSTA DO LLM:")
    print("-" * 50)
    print(resultado["llm"]["replies"][0])

    if usar_langfuse and "tracer" in resultado and resultado["tracer"].get("trace_url"):
        print(f"\nTrace no LangFuse: {resultado['tracer']['trace_url']}")


if __name__ == "__main__":
    main()
