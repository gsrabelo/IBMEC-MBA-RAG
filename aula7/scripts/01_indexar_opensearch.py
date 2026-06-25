"""
01_indexar_opensearch.py - Garante o indice do TCU (reaproveitado da Aula 4).

A Aula 7 REUSA os acordaos do TCU. Por padrao usa o indice 'aula4_hibrido' (criado
na Aula 4). Este script:
  - se o indice ja tem documentos -> apenas confirma (reaproveita, nao reindexa);
  - se estiver vazio (ou --recriar) -> indexa o corpus do TCU (gera embeddings Ollama).

Precisa de OpenSearch e Ollama.

Uso:
    python 01_indexar_opensearch.py                 # reaproveita se ja existir
    python 01_indexar_opensearch.py --recriar       # reindexa do zero
    python 01_indexar_opensearch.py --limite 300    # indexa so 300 docs (mais rapido)
"""

import argparse

import requests
from haystack.document_stores.types import DuplicatePolicy

import _comum


def main():
    parser = argparse.ArgumentParser(description="Garante/reusa o indice do TCU.")
    parser.add_argument("--indice", default=_comum.INDICE_TCU, help="indice no OpenSearch")
    parser.add_argument("--limite", type=int, default=0, help="quantos docs indexar se for criar (0 = todos)")
    parser.add_argument("--recriar", action="store_true", help="apaga e reindexa")
    args = parser.parse_args()

    _comum.carregar_env()
    os_cfg = _comum.config_opensearch()
    auth = (os_cfg["usuario"], os_cfg["senha"]) if os_cfg["usuario"] else None

    print("=" * 60)
    print("  INDICE DO TCU (reaproveitado) - Aula 7")
    print("=" * 60)
    print(f"Indice: {args.indice}")

    store = _comum.abrir_store(args.indice)
    try:
        atual = store.count_documents()
    except Exception:
        atual = 0

    if atual > 0 and not args.recriar:
        print(f"Indice ja tem {atual} documentos -> REAPROVEITANDO (use --recriar para refazer).")
        return

    if args.recriar:
        try:
            requests.delete(f"{os_cfg['url']}/{args.indice}", auth=auth, timeout=10)
            print("Indice apagado (--recriar).")
        except Exception as e:
            print(f"Aviso: nao consegui apagar: {e}")
        store = _comum.abrir_store(args.indice)

    docs = _comum.documentos_haystack(limite=args.limite)
    print(f"\nGerando embeddings de {len(docs)} acordaos via Ollama (pode demorar)...")
    docs_emb = _comum.doc_embedder().run(documents=docs)["documents"]
    qtd = store.write_documents(docs_emb, policy=DuplicatePolicy.OVERWRITE)
    print(f"Pronto! {qtd} documentos indexados em '{args.indice}'.")
    print("Agora rode: python 05_benchmark.py  (ele gera o benchmark e compara as tecnicas)")


if __name__ == "__main__":
    main()
