"""
02_busca_hibrida_rrf.py - Compara BM25 x Densa x Hibrida (RRF).

Tres formas de buscar a mesma pergunta no indice da Aula 4:
  - BM25   : busca por PALAVRAS (boa para termos exatos, numeros de lei, siglas)
  - Densa  : busca por SIGNIFICADO (embeddings/kNN; boa para sinonimos e parafrase)
  - Hibrida: combina as duas e funde os rankings com RRF (Reciprocal Rank Fusion)

RRF = cada documento ganha pontos por 1/(k + posicao) em cada lista; somando as
duas listas, quem aparece bem nas duas sobe. Costuma ser melhor que so uma.

Precisa do indice criado pelo 01 + OpenSearch + Ollama.

Uso:
    python 02_busca_hibrida_rrf.py --query "operacao de credito com garantia da Uniao"
    python 02_busca_hibrida_rrf.py --query "..." --top-k 5
"""

import argparse

import _comum

_comum.carregar_env()  # antes de importar haystack

from haystack import Pipeline                                                 # noqa: E402
from haystack_integrations.components.embedders.ollama import OllamaTextEmbedder  # noqa: E402
from haystack_integrations.components.retrievers.opensearch import (          # noqa: E402
    OpenSearchBM25Retriever,
    OpenSearchEmbeddingRetriever,
    OpenSearchHybridRetriever,
)
from haystack_integrations.document_stores.opensearch import OpenSearchDocumentStore  # noqa: E402

QUERY_EXEMPLO = "operacao de credito com garantia da Uniao"


def abrir_store(indice):
    base_url, modelo = _comum.config_ollama()
    os_cfg = _comum.config_opensearch()
    auth = (os_cfg["usuario"], os_cfg["senha"]) if os_cfg["usuario"] else None
    store = OpenSearchDocumentStore(
        hosts=os_cfg["url"], index=indice, embedding_dim=_comum.dimensao_do_modelo(modelo),
        http_auth=auth, use_ssl=False, verify_certs=False,
    )
    return store, base_url, modelo


def busca_bm25(store, query, k):
    return OpenSearchBM25Retriever(document_store=store, top_k=k).run(query=query)["documents"]


def busca_densa(store, base_url, modelo, query, k):
    pipe = Pipeline()
    pipe.add_component("embedder", OllamaTextEmbedder(model=modelo, url=base_url))
    pipe.add_component("retriever", OpenSearchEmbeddingRetriever(document_store=store, top_k=k))
    pipe.connect("embedder.embedding", "retriever.query_embedding")
    return pipe.run({"embedder": {"text": query}})["retriever"]["documents"]


def busca_hibrida(store, base_url, modelo, query, k):
    retriever = OpenSearchHybridRetriever(
        document_store=store,
        embedder=OllamaTextEmbedder(model=modelo, url=base_url),
        top_k_bm25=k, top_k_embedding=k, top_k=k,
        join_mode="reciprocal_rank_fusion",
    )
    return retriever.run(query=query)["documents"]


def mostrar(nome, docs, k):
    print(f"\n[{nome}]")
    print("-" * 50)
    for i, d in enumerate(docs[:k], 1):
        titulo = (d.meta.get("titulo") or d.content)[:65].replace("\n", " ")
        print(f"  #{i} [{(d.score or 0):.3f}] {titulo}")


def main():
    parser = argparse.ArgumentParser(description="Compara BM25 x Densa x Hibrida (RRF).")
    parser.add_argument("--query", default=QUERY_EXEMPLO, help="pergunta de busca")
    parser.add_argument("--indice", default="aula4_hibrido", help="indice no OpenSearch")
    parser.add_argument("--top-k", type=int, default=5, help="quantos resultados mostrar")
    args = parser.parse_args()

    print("=" * 60)
    print("  BM25 x DENSA x HIBRIDA (RRF) - Aula 4")
    print("=" * 60)
    print(f"Query: {args.query}")

    store, base_url, modelo = abrir_store(args.indice)
    mostrar("BM25 (palavras)", busca_bm25(store, args.query, args.top_k), args.top_k)
    mostrar("DENSA (significado)", busca_densa(store, base_url, modelo, args.query, args.top_k), args.top_k)
    mostrar("HIBRIDA (RRF)", busca_hibrida(store, base_url, modelo, args.query, args.top_k), args.top_k)

    print("\nDica: compare quem aparece no topo de cada lista. A hibrida costuma "
          "juntar o melhor das duas (termos exatos + significado).")


if __name__ == "__main__":
    main()
