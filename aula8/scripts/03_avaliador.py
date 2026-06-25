"""
03_avaliador.py - O avaliador de relevancia (LLM-as-Judge) isolado.

E o "grader" que esta no coracao do CRAG: recebe a pergunta + documentos recuperados
e retorna um score 0-1 por documento. Aqui ele aparece SOZINHO para voce entender a
peca antes de monta-la no pipeline (04_crag.py).

Mostra tambem a ROTA que o CRAG tomaria a partir do score medio:
  score >= 0.7 -> local | 0.3-0.7 -> fusao (local+web) | < 0.3 -> web

Uso:
    python 03_avaliador.py --pergunta "quando as contas sao julgadas irregulares?"
    python 03_avaliador.py --pergunta "decisoes do STF em 2024 sobre interceptacao" --top-k 4
"""

import argparse

import _comum

_comum.carregar_env()


def main():
    parser = argparse.ArgumentParser(description="Avaliador LLM-as-Judge isolado (Aula 8).")
    parser.add_argument("--pergunta", required=True)
    parser.add_argument("--indice", default=_comum.INDICE_TCU)
    parser.add_argument("--top-k", type=int, default=4)
    args = parser.parse_args()

    cliente, modelo = _comum.groq_client()

    print("=" * 60)
    print("  AVALIADOR DE RELEVANCIA (LLM-as-Judge) - Aula 8")
    print("=" * 60)
    print(f"Pergunta: {args.pergunta}\n")

    store = _comum.abrir_store(args.indice)
    pipe = _comum.montar_busca(store, args.top_k)
    documentos = _comum.buscar(pipe, args.pergunta)

    media, detalhes = _comum.avaliar_documentos(cliente, modelo, args.pergunta, documentos)
    for d, score, motivo in detalhes:
        print(f"- {d.meta.get('id_original'):>10}  score={score:.2f}  {motivo}")
        print(f"    {d.content[:120]}...")

    rota = _comum.decidir_rota(media)
    print("-" * 60)
    print(f"Score medio = {media:.2f}")
    print(f"Limiares: alto>={_comum.LIMITE_ALTO} | baixo>={_comum.LIMITE_BAIXO}")
    print(f"ROTA do CRAG = {rota.upper()}  "
          f"({'so local' if rota=='local' else 'local+web' if rota=='fusao' else 'so web'})")


if __name__ == "__main__":
    main()
