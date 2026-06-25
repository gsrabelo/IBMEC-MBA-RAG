"""
05_comparar_tecnicas.py - Compara Parent-Child x RAPTOR x HyDE com RAGAS.

Roda as perguntas GERADAS (01_gerar_dataset.py) nas TRES tecnicas e mede a
qualidade com as 4 metricas do RAGAS (juiz = Groq, embeddings = Ollama):
  faithfulness, answer_relevancy, context_recall, context_precision.

Mostra a media por tecnica e uma MATRIZ EXPLICAVEL tecnica x grupo-de-pergunta
('tecnica_ideal'): espera-se a diagonal mais alta (cada tecnica melhor no seu grupo).

Robustez:
  - o grupo de cada linha e identificado PELA PERGUNTA (nao por posicao), entao a
    matriz nunca perde/desalinha colunas mesmo se o RAGAS devolver menos linhas;
  - medias ignoram NaN;
  - juiz configuravel (--judge ou GROQ_JUDGE_MODEL no .env) e contextos truncados,
    para perguntas de contexto longo (RAPTOR) nao serem descartadas pelo juiz.

Precisa de OpenSearch, Ollama e Groq. Reaproveita os builders de 02/03/04
(construir_responder), recriando os indices a cada execucao. Obs.: o 02_parent_child
cria DOIS indices no OpenSearch - 'aula6_parent_child' (folhas, com embedding) e
'aula6_parent_child_arvore' (pais + folhas, p/ o auto-merging); este script aciona os
dois automaticamente via construir_responder(..., recriar=True).

Uso:
    python 05_comparar_tecnicas.py
    python 05_comparar_tecnicas.py --judge llama-3.3-70b-versatile --limite 9
"""

import argparse
import os
import warnings

import _comum

warnings.filterwarnings("ignore")
_comum.carregar_env()

from langchain_groq import ChatGroq                                            # noqa: E402
from langchain_ollama import OllamaEmbeddings                                  # noqa: E402
from ragas import EvaluationDataset, SingleTurnSample, evaluate                # noqa: E402
from ragas.embeddings import LangchainEmbeddingsWrapper                        # noqa: E402
from ragas.llms import LangchainLLMWrapper                                     # noqa: E402
from ragas.metrics import (                                                    # noqa: E402
    Faithfulness,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
    ResponseRelevancy,
)

# Limites para nao estourar o contexto do juiz (causa de linhas descartadas).
MAX_CTX = 5            # no maximo 5 trechos por pergunta
MAX_CTX_CHARS = 1500   # no maximo 1500 chars por trecho


def montar_dataset(responder, perguntas):
    amostras = []
    for p in perguntas:
        try:
            res = responder(p["pergunta"])
            resposta, contextos = res[0], res[1]
        except Exception as e:
            resposta, contextos = f"[ERRO: {e}]", []
        # trunca os contextos enviados ao juiz
        contextos = [c[:MAX_CTX_CHARS] for c in (contextos or [])[:MAX_CTX]] or ["(sem contexto)"]
        amostras.append(SingleTurnSample(
            user_input=p["pergunta"],
            retrieved_contexts=contextos,
            response=resposta,
            reference=p.get("ground_truth", p.get("resposta_esperada", "")),
        ))
    return EvaluationDataset(samples=amostras)


def avaliar_df(dataset, mapa_grupo, llm, emb):
    """Roda o RAGAS e devolve (df, colunas_metricas). O grupo vem PELA pergunta."""
    metricas = [Faithfulness(), ResponseRelevancy(), LLMContextRecall(),
                LLMContextPrecisionWithReference()]
    resultado = evaluate(dataset=dataset, metrics=metricas, llm=llm, embeddings=emb)
    df = resultado.to_pandas()
    cols = [c for c in df.columns if df[c].dtype.kind in "fc"]
    df["_composto"] = df[cols].mean(axis=1, skipna=True)
    # identifica o grupo de cada linha pela PERGUNTA (robusto a linhas faltando/reordenadas)
    df["_grupo"] = df["user_input"].map(mapa_grupo).fillna("?")
    return df, cols


def main():
    parser = argparse.ArgumentParser(description="Compara Parent-Child x RAPTOR x HyDE (RAGAS).")
    parser.add_argument("--limite", type=int, default=0, help="quantas perguntas usar (0 = todas)")
    parser.add_argument("--judge", default=None, help="modelo juiz da Groq (padrao: GROQ_JUDGE_MODEL ou o do .env)")
    args = parser.parse_args()

    perguntas = _comum.carregar_perguntas()
    if args.limite and args.limite > 0:
        perguntas = perguntas[: args.limite]
    # mapa pergunta -> grupo (tecnica_ideal) e lista de grupos existentes
    mapa_grupo = {p["pergunta"]: p.get("tecnica_ideal", "?") for p in perguntas}
    grupos_unicos = sorted(set(mapa_grupo.values()))

    from collections import Counter
    print("=" * 60)
    print("  COMPARACAO DE TECNICAS (RAGAS) - Aula 6")
    print("=" * 60)
    print(f"Perguntas: {len(perguntas)} | grupos: {dict(Counter(mapa_grupo.values()))}")

    base_url, modelo = _comum.config_ollama()
    _, groq_modelo, _ = _comum.config_groq()
    judge_model = args.judge or os.getenv("GROQ_JUDGE_MODEL", groq_modelo)
    print(f"Juiz (Groq): {judge_model} | embeddings: Ollama {modelo}")
    llm = LangchainLLMWrapper(ChatGroq(model=judge_model, temperature=0.0))
    emb = LangchainEmbeddingsWrapper(OllamaEmbeddings(model=modelo, base_url=base_url))

    print("\nConstruindo as 3 tecnicas (indexacao + clusters/resumos)...")
    pc = _comum.importar_script("02_parent_child.py").construir_responder("aula6_parent_child", recriar=True)
    rp = _comum.importar_script("03_raptor.py").construir_responder("aula6_raptor", recriar=True)
    hy = _comum.importar_script("04_hyde.py").construir_responder("aula6_hyde", recriar=True)
    tecnicas = {"Parent-Child": pc, "RAPTOR": rp, "HyDE": hy}

    medias, matriz, linhas = {}, {}, {}
    for nome, responder in tecnicas.items():
        print(f"\nAvaliando: {nome} (rodando as perguntas + RAGAS)...")
        dataset = montar_dataset(responder, perguntas)
        df, cols = avaliar_df(dataset, mapa_grupo, llm, emb)
        linhas[nome] = len(df)
        medias[nome] = {c: float(df[c].mean(skipna=True)) for c in cols}
        matriz[nome] = df.groupby("_grupo")["_composto"].mean().to_dict()

    # Tabela 1: medias gerais por tecnica.
    cols = sorted({c for m in medias.values() for c in m})
    print("\n" + "=" * 60)
    print("  MEDIAS GERAIS (por tecnica)")
    print("=" * 60)
    print(f"{'Tecnica':<14}" + "".join(f"{c[:15]:>17}" for c in cols))
    for nome, m in medias.items():
        print(f"{nome:<14}" + "".join(f"{m.get(c, 0):>17.3f}" for c in cols))

    # Tabela 2 (EXPLICAVEL): score composto por grupo de pergunta.
    print("\n" + "=" * 60)
    print("  SCORE COMPOSTO por TECNICA (linha) x GRUPO DE PERGUNTA (coluna)")
    print("  (espera-se a diagonal mais alta: cada tecnica melhor no seu grupo)")
    print("=" * 60)
    rotulo = "Tecnica/Grupo"
    print(f"{rotulo:<14}" + "".join(f"{g[:14]:>16}" for g in grupos_unicos))
    for nome in tecnicas:
        vals = matriz[nome]
        print(f"{nome:<14}" + "".join(f"{vals.get(g, float('nan')):>16.3f}" for g in grupos_unicos))

    # Aviso se o juiz descartou linhas (transparencia).
    print("\nLinhas avaliadas por tecnica (de", len(perguntas), "perguntas):", linhas)
    if any(v < len(perguntas) for v in linhas.values()):
        print("  [!] O juiz descartou algumas linhas. Tente --judge llama-3.3-70b-versatile.")
    print("Leitura: na coluna 'RAPTOR' (tematicas) espera-se o RAPTOR liderar; "
          "nas colunas Parent-Child/HyDE, as respectivas tecnicas.")


if __name__ == "__main__":
    main()
