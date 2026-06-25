"""
02_consultar_modos.py - Compara os 4 modos de busca do LightRAG.

Modos:
  - naive  : busca vetorial simples (sem grafo) = RAG convencional (baseline).
  - local  : foca em ENTIDADES e seus vizinhos no grafo (perguntas sobre pessoas/leis/casos).
  - global : usa SUMARIOS de comunidades (perguntas amplas/tematicas).
  - hybrid : combina local + global + chunks (melhor qualidade geral).

Roda a MESMA pergunta nos modos escolhidos para voce comparar as respostas.

Precisa do grafo construido antes: python 01_indexar_grafo.py

Uso:
    python 02_consultar_modos.py --pergunta "o que e colaboracao premiada e quem pode firmar?"
    python 02_consultar_modos.py --pergunta "..." --modos naive,hybrid
"""

import argparse
import asyncio

import _comum

_comum.carregar_env()

MODOS = ["naive", "local", "global", "hybrid"]


async def consultar(pergunta, modos):
    from lightrag import QueryParam

    rag = await _comum.criar_rag()
    try:
        for modo in modos:
            print("\n" + "=" * 60)
            print(f"  MODO: {modo}")
            print("=" * 60)
            try:
                resp = await rag.aquery(pergunta, param=QueryParam(mode=modo))
                print(resp)
            except Exception as e:
                print(f"[erro no modo {modo}] {e}")
    finally:
        await rag.finalize_storages()


def main():
    parser = argparse.ArgumentParser(description="Compara os modos de query do LightRAG (Aula 9).")
    parser.add_argument("--pergunta", required=True)
    parser.add_argument("--modos", default="naive,local,global,hybrid",
                        help="lista separada por virgula (padrao: todos)")
    args = parser.parse_args()

    modos = [m.strip() for m in args.modos.split(",") if m.strip() in MODOS]
    if not _comum.grafo_existe():
        print("[ATENCAO] grafo nao encontrado. Rode antes: python 01_indexar_grafo.py")
        return

    print("=" * 60)
    print("  CONSULTA NOS MODOS DO LIGHTRAG - Aula 9")
    print("=" * 60)
    print(f"Pergunta: {args.pergunta}")
    print(f"Modos: {modos}")
    asyncio.run(consultar(args.pergunta, modos))


if __name__ == "__main__":
    main()
