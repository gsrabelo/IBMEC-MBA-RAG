"""
03_ragas_avaliar.py - Avalia o RAG com as 4 metricas do RAGAS.

RAGAS usa um LLM como "juiz" (aqui a Groq) e embeddings (aqui o Ollama) para medir:
  - Faithfulness      : a resposta se apoia mesmo nos trechos? (anti-alucinacao)
  - Answer Relevancy  : a resposta responde a pergunta?
  - Context Recall    : os trechos cobrem o que o gabarito exige?
  - Context Precision : os trechos relevantes vieram no topo?

Le o dataset gerado pelo 01 e imprime as medias. Salva os resultados para o 04
enviar ao LangFuse.

Precisa do dataset (rode o 01 antes), Groq e Ollama.

Uso:
    python 03_ragas_avaliar.py
    python 03_ragas_avaliar.py --limite 10
"""

import argparse
import json
import warnings

import _comum

warnings.filterwarnings("ignore")
_comum.carregar_env()

from ragas import EvaluationDataset, SingleTurnSample, evaluate                # noqa: E402
from ragas.embeddings import LangchainEmbeddingsWrapper                        # noqa: E402
from ragas.llms import LangchainLLMWrapper                                     # noqa: E402
from ragas.metrics import (                                                    # noqa: E402
    Faithfulness,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
    ResponseRelevancy,
)


def main():
    parser = argparse.ArgumentParser(description="Avaliacao RAGAS (4 metricas).")
    parser.add_argument("--limite", type=int, default=0, help="avalia so os N primeiros pares (0 = todos)")
    args = parser.parse_args()

    dados = _comum.carregar_dataset_completo()
    if args.limite and args.limite > 0:
        dados = dados[: args.limite]

    print("=" * 60)
    print("  AVALIACAO RAGAS - Aula 5")
    print("=" * 60)
    print(f"Pares a avaliar: {len(dados)}")

    # Monta o dataset no formato do RAGAS.
    amostras = [
        SingleTurnSample(
            user_input=d["question"],
            retrieved_contexts=d.get("contexts", []) or ["(sem contexto)"],
            response=d.get("answer", ""),
            reference=d.get("ground_truth", ""),
        )
        for d in dados
    ]
    dataset = EvaluationDataset(samples=amostras)

    # Juiz (Groq) + embeddings (Ollama).
    chat, modelo = _comum.chat_groq(temperature=0.0)
    llm = LangchainLLMWrapper(chat)
    emb = LangchainEmbeddingsWrapper(_comum.ollama_embeddings_lc())
    print(f"Juiz LLM: Groq {modelo} | embeddings: Ollama")

    metricas = [Faithfulness(), ResponseRelevancy(), LLMContextRecall(),
                LLMContextPrecisionWithReference()]

    print("\nRodando avaliacao (cada metrica faz chamadas ao LLM; pode demorar)...")
    resultado = evaluate(dataset=dataset, metrics=metricas, llm=llm, embeddings=emb)

    # Medias por metrica (a partir do DataFrame de resultados).
    df = resultado.to_pandas()
    colunas_metricas = [c for c in df.columns
                        if df[c].dtype.kind in "fc" and c not in ("user_input", "response")]
    medias = {c: float(df[c].mean()) for c in colunas_metricas}

    print("\nMEDIAS RAGAS")
    print("-" * 40)
    for nome, valor in medias.items():
        print(f"  {nome:35} {valor:.3f}")

    with open(_comum.RESULTADOS_RAGAS, "w", encoding="utf-8") as f:
        json.dump({"medias": medias, "n_pares": len(dados)}, f, ensure_ascii=False, indent=2)
    print(f"\nResultados salvos em: {_comum.RESULTADOS_RAGAS}")
    print("Para registrar no LangFuse: python 05_langfuse_scores.py")


if __name__ == "__main__":
    main()
