"""
05_avaliar_busca.py - Avalia a busca: Densa vs Hibrida (MRR, Recall, NDCG).

Usa as queries de avaliacao (com gabarito de documentos relevantes) para medir,
de forma objetiva, qual estrategia recupera melhor:
  - MRR@k    : quao no topo veio o primeiro documento relevante
  - Recall@k : que fracao dos relevantes apareceu no top-k
  - NDCG@k   : qualidade do ranking (premia acertos no topo)

Compara DENSA (so kNN) com HIBRIDA (BM25 + densa + RRF) no mesmo indice.

Antes de rodar, indexe com:  python 01_indexar_hibrido.py
Precisa de OpenSearch + Ollama.

Uso:
    python 05_avaliar_busca.py
    python 05_avaliar_busca.py --top-k 10
"""

import argparse

import numpy as np

import _comum

_comum.carregar_env()  # antes de importar haystack

from haystack import Pipeline                                                 # noqa: E402
from haystack.components.joiners import DocumentJoiner                         # noqa: E402
from haystack_integrations.components.embedders.ollama import OllamaTextEmbedder  # noqa: E402
from haystack_integrations.components.retrievers.opensearch import (          # noqa: E402
    OpenSearchBM25Retriever,
    OpenSearchEmbeddingRetriever,
)
from haystack_integrations.document_stores.opensearch import OpenSearchDocumentStore  # noqa: E402


def mrr(ids, relevantes, k):
    for i, d in enumerate(ids[:k], 1):
        if d in relevantes:
            return 1.0 / i
    return 0.0


def recall(ids, relevantes, k):
    if not relevantes:
        return 0.0
    return len(set(ids[:k]) & set(relevantes)) / len(relevantes)


def ndcg(ids, relevantes, k):
    dcg = sum((1.0 if d in relevantes else 0.0) / np.log2(i + 1) for i, d in enumerate(ids[:k], 1))
    idcg = sum(1.0 / np.log2(i + 1) for i in range(1, min(len(relevantes), k) + 1))
    return dcg / idcg if idcg > 0 else 0.0


def ids_de(docs):
    return [d.meta.get("id_original") for d in docs]


def main():
    parser = argparse.ArgumentParser(description="Avalia Densa vs Hibrida.")
    parser.add_argument("--indice", default="aula4_hibrido", help="indice no OpenSearch")
    parser.add_argument("--top-k", type=int, default=10, help="K das metricas (padrao: 10)")
    args = parser.parse_args()

    base_url, modelo = _comum.config_ollama()
    os_cfg = _comum.config_opensearch()
    auth = (os_cfg["usuario"], os_cfg["senha"]) if os_cfg["usuario"] else None
    store = OpenSearchDocumentStore(
        hosts=os_cfg["url"], index=args.indice, embedding_dim=_comum.dimensao_do_modelo(modelo),
        http_auth=auth, use_ssl=False, verify_certs=False,
    )
    k = args.top_k

    # Pipeline DENSA: embedder -> embedding retriever
    densa = Pipeline()
    densa.add_component("embedder", OllamaTextEmbedder(model=modelo, url=base_url))
    densa.add_component("retriever", OpenSearchEmbeddingRetriever(document_store=store, top_k=k))
    densa.connect("embedder.embedding", "retriever.query_embedding")

    # Pipeline HIBRIDA: bm25 + densa -> RRF
    hibrida = Pipeline()
    hibrida.add_component("embedder", OllamaTextEmbedder(model=modelo, url=base_url))
    hibrida.add_component("bm25", OpenSearchBM25Retriever(document_store=store, top_k=k))
    hibrida.add_component("denso", OpenSearchEmbeddingRetriever(document_store=store, top_k=k))
    hibrida.add_component("juntar", DocumentJoiner(join_mode="reciprocal_rank_fusion", top_k=k))
    hibrida.connect("embedder.embedding", "denso.query_embedding")
    hibrida.connect("bm25.documents", "juntar.documents")
    hibrida.connect("denso.documents", "juntar.documents")

    queries = _comum.carregar_queries()
    print("=" * 60)
    print("  AVALIACAO DE BUSCA - Aula 4")
    print("=" * 60)
    print(f"Indice: {args.indice} | Queries: {len(queries)} | K={k}")
    print("Rodando buscas (pode demorar)...")

    soma = {"Densa": {"MRR": 0, "Recall": 0, "NDCG": 0},
            "Hibrida": {"MRR": 0, "Recall": 0, "NDCG": 0}}

    for q in queries:
        rel = q["documentos_relevantes"]
        ids_densa = ids_de(densa.run({"embedder": {"text": q["texto"]}})["retriever"]["documents"])
        ids_hibr = ids_de(hibrida.run(
            {"embedder": {"text": q["texto"]}, "bm25": {"query": q["texto"]}})["juntar"]["documents"])
        for nome, ids in [("Densa", ids_densa), ("Hibrida", ids_hibr)]:
            soma[nome]["MRR"] += mrr(ids, rel, k)
            soma[nome]["Recall"] += recall(ids, rel, k)
            soma[nome]["NDCG"] += ndcg(ids, rel, k)

    n = len(queries)
    print(f"\nMEDIAS (em {n} queries)")
    print("-" * 45)
    print(f"{'Estrategia':<12}{'MRR@'+str(k):>10}{'Recall@'+str(k):>12}{'NDCG@'+str(k):>10}")
    for nome in ["Densa", "Hibrida"]:
        s = soma[nome]
        print(f"{nome:<12}{s['MRR']/n:>10.3f}{s['Recall']/n:>12.3f}{s['NDCG']/n:>10.3f}")
    print("\nDica: normalmente a Hibrida supera a Densa pura, especialmente quando a "
          "query tem termos exatos (numeros, siglas, nomes de lei).")


if __name__ == "__main__":
    main()
