"""
01_indexar_grafo.py - Constroi o GRAFO DE CONHECIMENTO a partir do corpus (LightRAG).

O LightRAG le o corpus, faz chunking, pede ao LLM (Groq) para EXTRAIR entidades e
relacoes de cada trecho e monta um grafo (NetworkX) + indices vetoriais (NanoVectorDB),
tudo salvo em arquivo no working_dir. Essa etapa custa varias chamadas de LLM (e a
parte "cara" do Graph RAG), entao roda uma vez e fica persistido.

Uso:
    python 01_indexar_grafo.py
    python 01_indexar_grafo.py --recriar    # apaga o working_dir e reconstroi
"""

import argparse
import asyncio
import shutil

import _comum

_comum.carregar_env()


async def indexar(recriar):
    if recriar and _comum.WORKING_DIR.exists():
        print(f"--recriar: apagando {_comum.WORKING_DIR} ...")
        shutil.rmtree(_comum.WORKING_DIR, ignore_errors=True)

    if _comum.grafo_existe() and not recriar:
        print("Grafo ja existe (use --recriar para reconstruir). Nada a fazer.")
        return

    texto = _comum.ler_corpus()
    print(f"Corpus: {len(texto)} caracteres. Construindo o grafo (extracao via LLM)...")
    print("(pode demorar - o LLM extrai entidades/relacoes de cada chunk)\n")

    rag = await _comum.criar_rag()
    try:
        await rag.ainsert(texto)
    finally:
        await rag.finalize_storages()
    print("\nGrafo construido e salvo em:", _comum.WORKING_DIR)
    print("Explore com: python 04_explorar_grafo.py")


def main():
    parser = argparse.ArgumentParser(description="Indexa o grafo de conhecimento (Aula 9).")
    parser.add_argument("--recriar", action="store_true", help="apaga e reconstroi o grafo")
    args = parser.parse_args()

    print("=" * 60)
    print("  CONSTRUCAO DO GRAFO DE CONHECIMENTO - Aula 9 (LightRAG)")
    print("=" * 60)
    asyncio.run(indexar(args.recriar))


if __name__ == "__main__":
    main()
