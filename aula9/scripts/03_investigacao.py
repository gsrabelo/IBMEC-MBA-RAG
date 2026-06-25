"""
03_investigacao.py - Perguntas de INVESTIGACAO (multi-hop) sobre o grafo.

E aqui que o Graph RAG brilha: perguntas cuja resposta exige CONECTAR varias
entidades (multi-hop), algo que o RAG textual (naive) nao faz bem. Usamos o modo
'hybrid' (ou 'local') que navega pelo grafo.

Exemplos no contexto de crime organizado / colaboracao premiada do corpus:
  - conexoes entre uma entidade e organizacoes/operacoes;
  - quais leis/decisoes se relacionam a um instituto juridico;
  - panorama de uma operacao e seus envolvidos.

Precisa do grafo construido: python 01_indexar_grafo.py

Uso:
    python 03_investigacao.py                 # roda as perguntas de exemplo
    python 03_investigacao.py --pergunta "quais as conexoes entre o MPF e a colaboracao premiada?"
    python 03_investigacao.py --modo local
"""

import argparse
import asyncio

import _comum

_comum.carregar_env()

PERGUNTAS_EXEMPLO = [
    "Quais sao as conexoes entre o Ministerio Publico Federal e a colaboracao premiada?",
    "Que leis e decisoes se relacionam ao tema de organizacoes criminosas?",
    "Quem sao os atores (orgaos, tribunais) envolvidos nos casos do acervo e como se ligam?",
]


async def investigar(perguntas, modo):
    from lightrag import QueryParam

    rag = await _comum.criar_rag()
    try:
        for i, q in enumerate(perguntas, 1):
            print("\n" + "=" * 60)
            print(f"  [{i}] {q}   (modo: {modo})")
            print("=" * 60)
            try:
                resp = await rag.aquery(q, param=QueryParam(mode=modo))
                print(resp)
            except Exception as e:
                print(f"[erro] {e}")
    finally:
        await rag.finalize_storages()


def main():
    parser = argparse.ArgumentParser(description="Investigacao multi-hop no grafo (Aula 9).")
    parser.add_argument("--pergunta", default=None, help="pergunta unica (senao roda os exemplos)")
    parser.add_argument("--modo", default="hybrid", choices=["naive", "local", "global", "hybrid"])
    args = parser.parse_args()

    if not _comum.grafo_existe():
        print("[ATENCAO] grafo nao encontrado. Rode antes: python 01_indexar_grafo.py")
        return

    perguntas = [args.pergunta] if args.pergunta else PERGUNTAS_EXEMPLO
    print("=" * 60)
    print("  INVESTIGACAO MULTI-HOP (Graph RAG) - Aula 9")
    print("=" * 60)
    asyncio.run(investigar(perguntas, args.modo))


if __name__ == "__main__":
    main()
