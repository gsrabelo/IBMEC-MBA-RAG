"""
01_indexar.py - Indexa o corpus benchmark no OpenSearch (base das tecnicas).

Le os 30 documentos (com data/vigente/tipo), gera embeddings (Ollama) e indexa no
OpenSearch. Esse indice e usado pela busca densa (02 time-aware, 03 compressao, 04
comparacao com ColBERT). Os scripts 04/05/06 tambem montam seus proprios indices
especificos (ColBERT, CLIP) - este aqui e o indice de texto.

Uso:
    python 01_indexar.py
    python 01_indexar.py --recriar
"""

import argparse

import requests

import _comum

_comum.carregar_env()


def main():
    parser = argparse.ArgumentParser(description="Indexa o corpus da Aula 11.")
    parser.add_argument("--recriar", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print(f"  INDEXACAO - Aula 11  (indice '{_comum.INDICE}')")
    print("=" * 60)

    os_cfg = _comum.config_opensearch()
    auth = (os_cfg["usuario"], os_cfg["senha"]) if os_cfg["usuario"] else None
    if args.recriar:
        try:
            requests.delete(f"{os_cfg['url']}/{_comum.INDICE}", auth=auth, timeout=10)
            print("indice anterior removido (--recriar).")
        except Exception:
            pass

    store = _comum.abrir_store()
    if store.count_documents() > 0 and not args.recriar:
        print(f"Indice ja tem {store.count_documents()} documentos (use --recriar). Nada a fazer.")
        return

    docs = _comum.documentos_haystack()
    print(f"Gerando embeddings de {len(docs)} documentos (Ollama)...")
    embedder = _comum.doc_embedder()
    if hasattr(embedder, "warm_up"):
        embedder.warm_up()
    docs_emb = embedder.run(documents=docs)["documents"]
    store.write_documents(docs_emb)
    print(f"OK - {len(docs_emb)} documentos indexados em '{_comum.INDICE}'.")


if __name__ == "__main__":
    main()
