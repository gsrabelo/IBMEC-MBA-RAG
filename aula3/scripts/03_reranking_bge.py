"""
03_reranking_bge.py - Reranking com BGE-Reranker (Advanced RAG).

A busca vetorial (bi-encoder) e rapida, mas ordena por similaridade geral.
O reranker (cross-encoder BGE-reranker-v2-m3) le a pergunta JUNTO com cada
documento e da uma nota de relevancia mais precisa, reordenando os resultados.

Este script:
  1. Busca os top-N documentos no OpenSearch (ordem do bi-encoder).
  2. Reordena com o BGE-Reranker (ordem do cross-encoder).
  3. Mostra o ANTES e o DEPOIS, destacando o que subiu/desceu.

Precisa de OpenSearch + Ollama. Na 1a vez baixa o modelo do reranker (~560 MB).

Uso:
    python 03_reranking_bge.py --query "O suspeito pode ficar calado no interrogatorio?"
    python 03_reranking_bge.py --top-n 10 --top-k 5
"""

import argparse

import _comum

_comum.carregar_env()  # antes de importar haystack (necessario p/ tracing)

from haystack import Pipeline                                                 # noqa: E402
from haystack.components.rankers import TransformersSimilarityRanker          # noqa: E402
from haystack_integrations.components.embedders.ollama import OllamaTextEmbedder  # noqa: E402
from haystack_integrations.components.retrievers.opensearch import (          # noqa: E402
    OpenSearchEmbeddingRetriever,
)
from haystack_integrations.document_stores.opensearch import OpenSearchDocumentStore  # noqa: E402

MODELO_RERANKER = "BAAI/bge-reranker-v2-m3"
QUERY_EXEMPLO = "O suspeito pode ficar calado no interrogatorio?"


def buscar(query, indice, top_n):
    """Faz a busca vetorial e devolve os top-N documentos (ordem do bi-encoder)."""
    base_url, modelo = _comum.config_ollama()
    os_cfg = _comum.config_opensearch()
    auth = (os_cfg["usuario"], os_cfg["senha"]) if os_cfg["usuario"] else None
    store = OpenSearchDocumentStore(
        hosts=os_cfg["url"], index=indice, embedding_dim=_comum.dimensao_do_modelo(modelo),
        http_auth=auth, use_ssl=False, verify_certs=False,
    )
    pipe = Pipeline()
    pipe.add_component("embedder", OllamaTextEmbedder(model=modelo, url=base_url))
    pipe.add_component("retriever", OpenSearchEmbeddingRetriever(document_store=store, top_k=top_n))
    pipe.connect("embedder.embedding", "retriever.query_embedding")
    saida = pipe.run({"embedder": {"text": query}})
    return saida["retriever"]["documents"]


def rotulo(doc):
    return doc.meta.get("ementa", doc.content)[:60].replace("\n", " ")


def main():
    parser = argparse.ArgumentParser(description="Reranking com BGE-Reranker.")
    parser.add_argument("--query", default=QUERY_EXEMPLO, help="pergunta de busca")
    parser.add_argument("--indice", default="aula3_juridico", help="indice no OpenSearch")
    parser.add_argument("--top-n", type=int, default=10, help="quantos buscar antes do rerank")
    parser.add_argument("--top-k", type=int, default=5, help="quantos manter apos o rerank")
    args = parser.parse_args()

    print("=" * 60)
    print("  RERANKING COM BGE-RERANKER - Aula 3")
    print("=" * 60)
    print(f"Query: {args.query}")

    print(f"\nBuscando top-{args.top_n} no OpenSearch...")
    docs = buscar(args.query, args.indice, args.top_n)
    if not docs:
        print("Nenhum documento. Rode 02_indexar_opensearch.py antes.")
        return

    # Guarda a posicao original (antes do rerank) por id do documento.
    pos_original = {d.id: i + 1 for i, d in enumerate(docs)}

    print(f"Carregando o reranker {MODELO_RERANKER} (1a vez baixa ~560 MB)...")
    ranker = TransformersSimilarityRanker(model=MODELO_RERANKER, top_k=args.top_k)
    ranker.warm_up()
    rerankeados = ranker.run(query=args.query, documents=docs, top_k=args.top_k)["documents"]

    print("\nANTES (busca vetorial):")
    print("-" * 50)
    for i, d in enumerate(docs[:args.top_k], 1):
        print(f"  #{i} [{d.score:.3f}] {rotulo(d)}")

    print("\nDEPOIS (reranker BGE):")
    print("-" * 50)
    for i, d in enumerate(rerankeados, 1):
        antes = pos_original.get(d.id, "?")
        mov = (antes - i) if isinstance(antes, int) else 0
        seta = f"subiu {mov}" if mov > 0 else (f"desceu {abs(mov)}" if mov < 0 else "manteve")
        print(f"  #{i} [{d.score:.3f}] ({seta}, era #{antes}) {rotulo(d)}")

    print("\nDica: o reranker costuma trazer para o topo o documento que responde "
          "mais diretamente a pergunta.")


if __name__ == "__main__":
    main()
