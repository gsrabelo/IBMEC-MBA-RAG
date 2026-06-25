"""
05_avaliar_custo.py - Distribuicao de rotas (Adaptive) + analise de custo do agente.

Atende o criterio de avaliacao da aula: "quanto custaria rodar o agente 1.000 vezes?".
Faz duas coisas:
  (1) Classifica um conjunto de perguntas de exemplo (Adaptive RAG) e mostra a
      DISTRIBUICAO de rotas (sem_retrieval / simples / complexa).
  (2) Estima o CUSTO por rota e o custo de 1.000 execucoes - comparando "tudo agente"
      vs. "adaptativo" (so paga o caminho caro quando precisa).

Os numeros de tokens sao ESTIMATIVAS didaticas (ajuste --preco-in/--preco-out e os
tokens por rota conforme seu modelo). Preco padrao ~ Groq llama-3.3-70b-versatile.

Uso:
    python 05_avaliar_custo.py
    python 05_avaliar_custo.py --execucoes 1000 --preco-in 0.59 --preco-out 0.79
"""

import argparse

import _comum

_comum.carregar_env()

PERGUNTAS = [
    "O que e habeas corpus?",
    "O que significa transito em julgado?",
    "Qual a pena prevista no art. 2 da Lei 12.850/2013?",
    "Quantas ocorrencias de estelionato existem em SP no banco?",
    "Qual a ementa do acordao HC 127483?",
    "Compare os precedentes sobre prisao preventiva em crimes financeiros e identifique divergencias recentes.",
    "Quais leis sobre lavagem de dinheiro estao vigentes e o que a doutrina diz sobre tipologias?",
    "Qual o valor total envolvido nas ocorrencias de corrupcao no banco?",
]

# tokens medios ESTIMADOS por rota (entrada/saida). Ajuste conforme medir no LangFuse.
TOKENS_ROTA = {
    "sem_retrieval": (400, 300),    # 1 chamada LLM
    "simples":       (1500, 400),   # busca + 1 chamada LLM com contexto
    "complexa":      (6000, 1200),  # agente: varias chamadas LLM + observacoes
}


def custo_rota(rota, preco_in, preco_out):
    ti, to = TOKENS_ROTA[rota]
    return (ti / 1_000_000) * preco_in + (to / 1_000_000) * preco_out


def main():
    parser = argparse.ArgumentParser(description="Distribuicao de rotas + custo (Aula 10).")
    parser.add_argument("--execucoes", type=int, default=1000)
    parser.add_argument("--preco-in", type=float, default=0.59, help="USD por 1M tokens de entrada")
    parser.add_argument("--preco-out", type=float, default=0.79, help="USD por 1M tokens de saida")
    args = parser.parse_args()

    print("=" * 60)
    print("  DISTRIBUICAO DE ROTAS + CUSTO - Aula 10")
    print("=" * 60)

    cliente, modelo = _comum.groq_client()
    from collections import Counter
    dist = Counter()
    print(f"Classificando {len(PERGUNTAS)} perguntas de exemplo (modelo {modelo})...\n")
    # reutiliza o classificador do script 04 (nao duplica a logica)
    m04 = _comum_importar("04_adaptive_rag.py")
    for q in PERGUNTAS:
        classe = m04.classificar(cliente, modelo, q)
        dist[classe] += 1
        print(f"  [{classe:13}] {q}")

    n = sum(dist.values())
    print("\nDistribuicao de rotas:")
    for rota in ("sem_retrieval", "simples", "complexa"):
        print(f"  {rota:13}: {dist[rota]}/{n} ({100*dist[rota]/n:.0f}%)")

    print("\n" + "=" * 60)
    print(f"  CUSTO ESTIMADO para {args.execucoes} execucoes")
    print("=" * 60)
    print(f"(preco: ${args.preco_in}/1M in, ${args.preco_out}/1M out - tokens estimados por rota)\n")
    print(f"{'Rota':<14}{'US$/exec':>12}{'US$/'+str(args.execucoes):>14}")
    for rota in ("sem_retrieval", "simples", "complexa"):
        c = custo_rota(rota, args.preco_in, args.preco_out)
        print(f"{rota:<14}{c:>12.5f}{c*args.execucoes:>14.2f}")

    # tudo-agente vs adaptativo (usando a distribuicao medida)
    custo_tudo_agente = custo_rota("complexa", args.preco_in, args.preco_out) * args.execucoes
    custo_adaptativo = sum(
        (dist[r] / n) * custo_rota(r, args.preco_in, args.preco_out) for r in dist) * args.execucoes
    print("\nComparacao para", args.execucoes, "execucoes (com a distribuicao acima):")
    print(f"  Tudo-agente (sempre complexa): US$ {custo_tudo_agente:.2f}")
    print(f"  Adaptive RAG (rota por complexidade): US$ {custo_adaptativo:.2f}")
    economia = 100 * (1 - custo_adaptativo / custo_tudo_agente) if custo_tudo_agente else 0
    print(f"  Economia do Adaptive RAG: {economia:.0f}%")
    print("\nNota: tokens sao estimativas didaticas. Meca no LangFuse e ajuste TOKENS_ROTA.")


def _comum_importar(nome):
    import importlib.util
    caminho = _comum.PASTA_SCRIPTS / nome
    spec = importlib.util.spec_from_file_location(nome.replace(".py", ""), caminho)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


if __name__ == "__main__":
    main()
