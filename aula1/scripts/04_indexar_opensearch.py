"""
04_indexar_opensearch.py - Indexa o corpus juridico no OpenSearch (busca vetorial).

Passos:
  1. Le o corpus juridico.
  2. Gera o embedding de cada documento usando o Ollama.
  3. Grava os documentos + vetores em um indice kNN do OpenSearch.

Depois disso, o script 05_rag_minimo.py consegue buscar nesses documentos.

Precisa do OpenSearch e do Ollama rodando (veja 00_check_ambiente.py).

Uso:
    python 04_indexar_opensearch.py
    python 04_indexar_opensearch.py --indice aula1_juridico --recriar
"""

import argparse

import requests
from haystack import Document
from haystack.document_stores.types import DuplicatePolicy
from haystack_integrations.components.embedders.ollama import OllamaDocumentEmbedder
from haystack_integrations.document_stores.opensearch import OpenSearchDocumentStore

import _comum


def apagar_indice(url, indice, auth):
    """Apaga o indice no OpenSearch, se ele existir (usado com --recriar)."""
    try:
        r = requests.delete(f"{url}/{indice}", auth=auth, timeout=10)
        if r.status_code in (200, 404):
            print(f"Indice '{indice}' apagado (ou ja nao existia).")
    except Exception as e:
        print(f"Aviso: nao consegui apagar o indice: {e}")


def main():
    parser = argparse.ArgumentParser(description="Indexa o corpus no OpenSearch.")
    parser.add_argument("--indice", default="aula1_juridico", help="nome do indice")
    parser.add_argument("--modelo", default=None, help="modelo de embedding (padrao: .env)")
    parser.add_argument("--recriar", action="store_true",
                        help="apaga o indice antes de indexar (comeca do zero)")
    args = parser.parse_args()

    _comum.carregar_env()
    base_url, modelo_padrao = _comum.config_ollama()
    modelo = args.modelo or modelo_padrao
    os_cfg = _comum.config_opensearch()
    auth = (os_cfg["usuario"], os_cfg["senha"]) if os_cfg["usuario"] else None
    dim = _comum.dimensao_do_modelo(modelo)

    print("=" * 60)
    print("  INDEXACAO NO OPENSEARCH - Aula 1")
    print("=" * 60)
    print(f"Indice: {args.indice} | Modelo: {modelo} | Dimensao: {dim}")

    if args.recriar:
        apagar_indice(os_cfg["url"], args.indice, auth)

    # Cria/abre o indice vetorial no OpenSearch.
    store = OpenSearchDocumentStore(
        hosts=os_cfg["url"],
        index=args.indice,
        embedding_dim=dim,
        http_auth=auth,
        use_ssl=False,
        verify_certs=False,
    )

    # Transforma cada documento do corpus em um Document do Haystack.
    corpus = _comum.carregar_corpus()
    documentos = [
        Document(
            content=d["texto"],
            meta={
                "id_original": d["id"],
                "titulo": d["titulo"],
                "categoria": d["categoria"],
            },
        )
        for d in corpus
    ]

    print(f"\nGerando embeddings de {len(documentos)} documentos via Ollama...")
    embedder = OllamaDocumentEmbedder(model=modelo, url=base_url)
    documentos_com_vetor = embedder.run(documents=documentos)["documents"]

    print("Gravando no OpenSearch...")
    qtd = store.write_documents(documentos_com_vetor, policy=DuplicatePolicy.OVERWRITE)
    print(f"\nPronto! {qtd} documentos indexados no indice '{args.indice}'.")
    print("Agora rode: python 05_rag_minimo.py --pergunta \"sua pergunta aqui\"")


if __name__ == "__main__":
    main()
