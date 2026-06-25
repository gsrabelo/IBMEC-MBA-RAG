"""
01_indexar_hibrido.py - Indexa o corpus da Aula 4 para BUSCA HIBRIDA.

Busca hibrida = combinar busca por palavras (BM25, no campo de texto) com busca
por significado (kNN, no campo de embedding). Para isso, cada documento precisa
ter os DOIS no mesmo indice:
  - o texto (o OpenSearch indexa para BM25 automaticamente)
  - o vetor/embedding (gerado aqui com o Ollama, para a busca kNN)

Este script le o corpus (1100 acordaos do TCU), gera os embeddings e grava tudo
em um indice do OpenSearch que serve para BM25 e para kNN ao mesmo tempo.

Precisa de OpenSearch e Ollama (veja 00_check_ambiente.py).

Uso:
    python 01_indexar_hibrido.py
    python 01_indexar_hibrido.py --recriar
    python 01_indexar_hibrido.py --limite 200      # so os 200 primeiros (teste rapido)
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
    parser = argparse.ArgumentParser(description="Indexa o corpus da Aula 4 (hibrido).")
    parser.add_argument("--indice", default="aula4_hibrido", help="nome do indice")
    parser.add_argument("--modelo", default=None, help="modelo de embedding (padrao: .env)")
    parser.add_argument("--limite", type=int, default=0, help="indexa so os N primeiros (0 = todos)")
    parser.add_argument("--recriar", action="store_true", help="apaga o indice antes de indexar")
    args = parser.parse_args()

    _comum.carregar_env()
    base_url, modelo_padrao = _comum.config_ollama()
    modelo = args.modelo or modelo_padrao
    os_cfg = _comum.config_opensearch()
    auth = (os_cfg["usuario"], os_cfg["senha"]) if os_cfg["usuario"] else None
    dim = _comum.dimensao_do_modelo(modelo)

    print("=" * 60)
    print("  INDEXACAO HIBRIDA NO OPENSEARCH - Aula 4")
    print("=" * 60)
    print(f"Indice: {args.indice} | Modelo: {modelo} (dim {dim})")

    if args.recriar:
        apagar_indice(os_cfg["url"], args.indice, auth)

    store = OpenSearchDocumentStore(
        hosts=os_cfg["url"], index=args.indice, embedding_dim=dim,
        http_auth=auth, use_ssl=False, verify_certs=False,
    )

    docs_corpus = _comum.carregar_corpus(limite=args.limite)
    documentos = [
        Document(
            content=d["texto"],
            meta={"id_original": d["id"], "tipo": d.get("tipo", ""),
                  "titulo": d.get("titulo", "")},
        )
        for d in docs_corpus
    ]

    print(f"\nGerando embeddings de {len(documentos)} documentos via Ollama (pode demorar)...")
    embedder = OllamaDocumentEmbedder(model=modelo, url=base_url)
    docs_com_vetor = embedder.run(documents=documentos)["documents"]

    print("Gravando no OpenSearch (texto p/ BM25 + vetor p/ kNN)...")
    qtd = store.write_documents(docs_com_vetor, policy=DuplicatePolicy.OVERWRITE)
    print(f"\nPronto! {qtd} documentos indexados no indice '{args.indice}'.")
    print("Agora rode: python 02_busca_hibrida_rrf.py  ou  python 04_hybrid_rag.py")


if __name__ == "__main__":
    main()
