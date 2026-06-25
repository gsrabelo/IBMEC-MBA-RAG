"""
01_indexar_opensearch.py - Garante/reaproveita o indice do TCU para a Aula 8.

A Aula 8 NAO cria corpus proprio: reaproveita os acordaos do TCU ja indexados na
Aula 4 (indice 'aula4_hibrido'). Este script so reindexa se o indice estiver vazio
(ou se voce passar --recriar).

Uso:
    python 01_indexar_opensearch.py            # reaproveita se ja tiver documentos
    python 01_indexar_opensearch.py --recriar  # apaga e reindexa o corpus do TCU
    python 01_indexar_opensearch.py --limite 50
"""

import argparse

import _comum

_comum.carregar_env()


def main():
    parser = argparse.ArgumentParser(description="Indexa/reaproveita o indice do TCU (Aula 8).")
    parser.add_argument("--indice", default=_comum.INDICE_TCU)
    parser.add_argument("--recriar", action="store_true", help="apaga e reindexa")
    parser.add_argument("--limite", type=int, default=0, help="limita nº de docs (0 = todos)")
    args = parser.parse_args()

    print("=" * 60)
    print(f"  INDEXACAO / REAPROVEITAMENTO - indice '{args.indice}'")
    print("=" * 60)

    store = _comum.abrir_store(args.indice)
    try:
        n = store.count_documents()
    except Exception as e:
        print(f"[ERRO] nao consegui acessar o OpenSearch: {e}")
        return

    if n > 0 and not args.recriar:
        print(f"Indice ja tem {n} documentos. Reaproveitando (use --recriar para reindexar).")
        return

    if args.recriar and n > 0:
        print(f"--recriar: removendo {n} documentos existentes...")
        try:
            store.client.indices.delete(index=args.indice, ignore=[400, 404])
        except Exception:
            pass
        store = _comum.abrir_store(args.indice)

    print("Lendo corpus do TCU (Aula 4) e gerando embeddings com Ollama...")
    docs = _comum.documentos_haystack(limite=args.limite)
    embedder = _comum.doc_embedder()
    if hasattr(embedder, "warm_up"):
        embedder.warm_up()
    docs_emb = embedder.run(documents=docs)["documents"]
    store.write_documents(docs_emb)
    print(f"OK - {len(docs_emb)} documentos indexados em '{args.indice}'.")


if __name__ == "__main__":
    main()
