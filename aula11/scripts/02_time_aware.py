"""
02_time_aware.py - Time-Aware RAG: relevancia ponderada por recencia.

No Direito, lei revogada/desatualizada e risco. Aqui recuperamos por similaridade e
RE-RANQUEAMOS multiplicando o score pela funcao de DECAY temporal:

    decay(idade) = exp( -ln(2) * max(0, idade_dias - offset) / scale )
    score_final  = score_relevancia * decay(idade)

Mostra o ranking SEM e COM decay (a ordem muda priorizando documentos recentes) e o
filtro opcional de vigencia. E a versao didatica do function_score/exp do OpenSearch.

Precisa do indice indexado: python 01_indexar.py

Uso:
    python 02_time_aware.py --pergunta "regras de licitacao"
    python 02_time_aware.py --pergunta "..." --scale 365 --offset 30 --top-k 8
    python 02_time_aware.py --pergunta "..." --so-vigentes
"""

import argparse
import math
from datetime import date

import _comum

_comum.carregar_env()


def idade_dias(data_str):
    try:
        a, m, d = (int(x) for x in data_str.split("-"))
        return max(0, (date.today() - date(a, m, d)).days)
    except Exception:
        return 0


def decay(idade, scale, offset):
    return math.exp(-math.log(2) * max(0, idade - offset) / max(1, scale))


def main():
    parser = argparse.ArgumentParser(description="Time-Aware RAG (decay temporal).")
    parser.add_argument("--pergunta", required=True)
    parser.add_argument("--scale", type=int, default=3650,
                        help="meia-vida do decay em dias (padrao 3650=10 anos; ajuste ao corpus)")
    parser.add_argument("--offset", type=int, default=365, help="periodo de graca (dias)")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--so-vigentes", action="store_true", help="filtra documentos nao vigentes")
    args = parser.parse_args()

    store = _comum.abrir_store()
    if store.count_documents() == 0:
        print("[ATENCAO] indice vazio. Rode antes: python 01_indexar.py")
        return
    pipe = _comum.montar_busca(store, args.top_k)
    docs = _comum.buscar(pipe, args.pergunta)
    if args.so_vigentes:
        docs = [d for d in docs if d.meta.get("vigente", True)]

    print("=" * 70)
    print("  TIME-AWARE RAG - Aula 11")
    print("=" * 70)
    print(f"Pergunta: {args.pergunta} | scale={args.scale}d offset={args.offset}d "
          f"| so_vigentes={args.so_vigentes}\n")

    linhas = []
    for d in docs:
        rel = d.score or 0.0
        dias = idade_dias(d.meta.get("data", ""))
        dec = decay(dias, args.scale, args.offset)
        linhas.append((d, rel, dias, dec, rel * dec))

    print("SEM decay (so relevancia):")
    for d, rel, dias, dec, fin in sorted(linhas, key=lambda x: x[1], reverse=True)[:args.top_k]:
        print(f"  rel={rel:.3f}  {d.meta.get('data')}  {d.meta.get('id_original')}  {d.content[:55]}")

    print("\nCOM decay (relevancia x recencia):")
    for d, rel, dias, dec, fin in sorted(linhas, key=lambda x: x[4], reverse=True)[:args.top_k]:
        print(f"  final={fin:.3f} (rel={rel:.2f} x decay={dec:.2f}, {dias}d)  "
              f"{d.meta.get('data')}  {d.meta.get('id_original')}  {d.content[:45]}")

    print("\nLeitura: o decay empurra documentos antigos para baixo. Ajuste --scale "
          "(menor = mais agressivo) conforme o dominio. No OpenSearch nativo, o "
          "equivalente e function_score com 'exp' no campo de data.")


if __name__ == "__main__":
    main()
