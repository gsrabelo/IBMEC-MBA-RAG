"""
02_indexar_opensearch.py - Indexa o corpus da Aula 3 no OpenSearch (busca vetorial).

Passos:
  1. Le o corpus (datasets/corpus_juridico_aula3.json, ~80 documentos).
  2. Gera o embedding de cada documento (ementa + texto) com o Ollama.
  3. Grava os documentos + vetores em um indice kNN do OpenSearch.

Depois disso, os scripts 03 (reranking) e 04 (Advanced RAG) conseguem buscar.

Precisa de OpenSearch e Ollama rodando (veja 00_check_ambiente.py).

Uso:
    python 02_indexar_opensearch.py
    python 02_indexar_opensearch.py --indice aula3_juridico --recriar
"""

import argparse

import requests
from haystack import Document
from haystack.document_stores.types import DuplicatePolicy
from haystack_integrations.components.embedders.ollama import OllamaDocumentEmbedder
from haystack_integrations.document_stores.opensearch import OpenSearchDocumentStore

import _comum


def apagar_indice(url, indice, auth):
    try:
        r = requests.delete(f"{url}/{indice}", auth=auth, timeout=10)
        if r.status_code in (200, 404):
            print(f"Indice '{indice}' apagado (ou ja nao existia).")
    except Exception as e:
        print(f"Aviso: nao consegui apagar o indice: {e}")


def main():
    parser = argparse.ArgumentParser(description="Indexa o corpus da Aula 3 no OpenSearch.")
    parser.add_argument("--indice", default="aula3_juridico", help="nome do indice")
    parser.add_argument("--modelo", default=None, help="modelo de embedding (padrao: .env)")
    parser.add_argument("--recriar", action="store_true", help="apaga o indice antes de indexar")
    args = parser.parse_args()

    _comum.carregar_env()
    base_url, modelo_padrao = _comum.config_ollama()
    modelo = args.modelo or modelo_padrao
    os_cfg = _comum.config_opensearch()
    auth = (os_cfg["usuario"], os_cfg["senha"]) if os_cfg["usuario"] else None
    dim = _comum.dimensao_do_modelo(modelo)

    print("=" * 60)
    print("  INDEXACAO NO OPENSEARCH - Aula 3")
    print("=" * 60)
    print(f"Indice: {args.indice} | Modelo: {modelo} (dim {dim})")

    if args.recriar:
        apagar_indice(os_cfg["url"], args.indice, auth)

    store = OpenSearchDocumentStore(
        hosts=os_cfg["url"], index=args.indice, embedding_dim=dim,
        http_auth=auth, use_ssl=False, verify_certs=False,
    )

    documentos, _ = _comum.carregar_corpus()
    docs = [
        Document(
            content=d["texto_completo"],
            meta={"id_original": d["id"], "tipo": d.get("tipo", ""),
                  "tribunal": d.get("tribunal", ""), "ementa": d.get("ementa", "")[:200]},
        )
        for d in documentos
    ]

    print(f"\nGerando embeddings de {len(docs)} documentos via Ollama...")
    embedder = OllamaDocumentEmbedder(model=modelo, url=base_url)
    docs_com_vetor = embedder.run(documents=docs)["documents"]

    print("Gravando no OpenSearch...")
    qtd = store.write_documents(docs_com_vetor, policy=DuplicatePolicy.OVERWRITE)
    print(f"\nPronto! {qtd} documentos indexados no indice '{args.indice}'.")
    print("Agora rode: python 03_reranking_bge.py  ou  python 04_advanced_rag.py")


if __name__ == "__main__":
    main()
