"""
04_deepeval_testes.py - Testes de qualidade com DeepEval (judge = Groq).

DeepEval funciona como "testes unitarios" para RAG: cada metrica tem um limite
(threshold) e diz PASS/FAIL. Usamos um juiz LLM (Groq via LangChain). Metricas:
  - Faithfulness    : a resposta se apoia nos trechos (alto = bom)
  - AnswerRelevancy : a resposta responde a pergunta (alto = bom)
  - Hallucination   : inventou fora do contexto (baixo = bom)
  - Toxicity        : conteudo toxico (baixo = bom)
  - Bias            : vies (baixo = bom)

Le o dataset gerado pelo 01. Por ser lento, avalia poucos casos por padrao.

Precisa do dataset (rode o 01 antes) e da Groq.

Uso:
    python 04_deepeval_testes.py
    python 04_deepeval_testes.py --limite 3
"""

import argparse
import warnings

import _comum

warnings.filterwarnings("ignore")
_comum.carregar_env()

from deepeval.metrics import (                                                 # noqa: E402
    AnswerRelevancyMetric,
    BiasMetric,
    FaithfulnessMetric,
    HallucinationMetric,
    ToxicityMetric,
)
from deepeval.models import DeepEvalBaseLLM                                     # noqa: E402
from deepeval.test_case import LLMTestCase                                      # noqa: E402
from langchain_core.messages import HumanMessage                               # noqa: E402


class JuizLangchain(DeepEvalBaseLLM):
    """Adapta um ChatModel do LangChain (Groq) para ser o juiz do DeepEval."""

    def __init__(self, modelo_lc, nome):
        self.modelo_lc = modelo_lc
        self._nome = nome

    def load_model(self):
        return self.modelo_lc

    def generate(self, prompt):
        return self.modelo_lc.invoke([HumanMessage(content=prompt)]).content

    async def a_generate(self, prompt):
        return self.generate(prompt)

    def get_model_name(self):
        return self._nome


def main():
    parser = argparse.ArgumentParser(description="Testes de qualidade com DeepEval.")
    parser.add_argument("--limite", type=int, default=5, help="quantos casos avaliar (padrao: 5)")
    args = parser.parse_args()

    dados = _comum.carregar_dataset_completo()[: args.limite]
    chat, modelo = _comum.chat_groq(temperature=0.0)
    juiz = JuizLangchain(chat, f"groq/{modelo}")

    print("=" * 60)
    print("  DEEPEVAL - Aula 5")
    print("=" * 60)
    print(f"Juiz: {juiz.get_model_name()} | Casos: {len(dados)}")

    for n, d in enumerate(dados, 1):
        print(f"\n--- Caso {n}: {d['question'][:55]}... ---")
        caso = LLMTestCase(
            input=d["question"], actual_output=d.get("answer", ""),
            retrieval_context=d.get("contexts", []) or ["(sem contexto)"],
            expected_output=d.get("ground_truth", ""),
        )
        caso_hall = LLMTestCase(
            input=d["question"], actual_output=d.get("answer", ""),
            context=d.get("contexts", []) or ["(sem contexto)"],
        )
        metricas = [
            ("Faithfulness", FaithfulnessMetric(threshold=0.80, model=juiz), caso, True),
            ("AnswerRelevancy", AnswerRelevancyMetric(threshold=0.75, model=juiz), caso, True),
            ("Hallucination", HallucinationMetric(threshold=0.20, model=juiz), caso_hall, False),
            ("Toxicity", ToxicityMetric(threshold=0.10, model=juiz), caso, False),
            ("Bias", BiasMetric(threshold=0.20, model=juiz), caso, False),
        ]
        for nome, metrica, alvo, maior_melhor in metricas:
            try:
                metrica.measure(alvo)
                status = "PASS" if metrica.is_successful() else "FAIL"
                seta = "(alto=bom)" if maior_melhor else "(baixo=bom)"
                print(f"  {nome:16} {metrica.score:.3f} {seta:12} -> {status}")
            except Exception as e:
                print(f"  {nome:16} erro: {str(e)[:60]}")

    print("\nDica: Faithfulness/AnswerRelevancy altos e Hallucination/Toxicity/Bias "
          "baixos = pipeline saudavel. FAIL aponta onde investigar.")


if __name__ == "__main__":
    main()
