"""
01_gerar_dataset.py - Gera um dataset balanceado para comparar as 3 tecnicas.

O corpus original da aula (10 docs curtos) nao exercita bem Parent-Child (precisa
de docs longos) nem RAPTOR (precisa de varios docs por tema). Aqui montamos um
dataset melhor a partir dos ACORDAOS DO TCU da Aula 4 (docs longos e numerosos):

  1. Amostra N documentos do corpus do TCU  -> salva como corpus_trabalho.json
     (e este corpus que as 3 tecnicas vao indexar - mesmo conteudo do indice aula4).
  2. Gera perguntas dos 3 tipos, cada uma marcada com a tecnica que favorece:
       - Parent-Child : pergunta ESPECIFICA sobre um detalhe de um doc longo
       - HyDE         : pergunta COLOQUIAL/leiga (gap de vocabulario)
       - RAPTOR       : pergunta AMPLA/TEMATICA que exige juntar varios docs (clusters)
     -> salva como perguntas_geradas.json (com documentos_relevantes e justificativa).

Assim a comparacao (05) fica eficiente e EXPLICAVEL: da para ver cada tecnica
brilhando no tipo de pergunta para o qual ela foi feita.

Precisa de Ollama (embeddings p/ clusterizar) e Groq (gerar perguntas).

Uso:
    python 01_gerar_dataset.py
    python 01_gerar_dataset.py --n-docs 80 --por-tecnica 6 --seed 42
"""

import argparse
import json
import random
import re

import numpy as np
from sklearn.cluster import KMeans

import _comum

PROMPT_ESPECIFICA = (
    "Com base no acordao do TCU abaixo, gere UMA pergunta ESPECIFICA e factual sobre "
    "um detalhe do documento (valor, artigo, orgao, processo, decisao) e a resposta "
    "correta extraida do texto. Responda em JSON: {{\"pergunta\": \"...\", \"resposta\": \"...\"}}\n\n{texto}"
)
PROMPT_COLOQUIAL = (
    "Com base no acordao do TCU abaixo, gere UMA pergunta em LINGUAGEM COLOQUIAL/leiga "
    "(como um cidadao comum perguntaria, SEM jargao juridico) cuja resposta esta no "
    "documento, e a resposta correta em linguagem tecnica. "
    "Responda em JSON: {{\"pergunta\": \"...\", \"resposta\": \"...\"}}\n\n{texto}"
)
PROMPT_TEMATICA = (
    "Os trechos abaixo sao de varios acordaos do TCU sobre um tema parecido. Gere UMA "
    "pergunta AMPLA/TEMATICA que so possa ser bem respondida SINTETIZANDO varios deles, "
    "e uma resposta que resume o panorama. "
    "Responda em JSON: {{\"pergunta\": \"...\", \"resposta\": \"...\"}}\n\n{trechos}"
)

JUSTIFICATIVA = {
    "Parent-Child": "Pergunta especifica sobre um detalhe de um documento longo: a busca precisa do trecho exato (filho) e a resposta precisa do contexto ao redor (pai).",
    "HyDE": "Pergunta coloquial: ha gap de vocabulario com o documento tecnico; o documento hipotetico aproxima a busca do corpus.",
    "RAPTOR": "Pergunta ampla/tematica que exige sintetizar varios documentos do mesmo tema (os resumos de cluster ajudam).",
}


def extrair_json(texto):
    try:
        return json.loads(texto)
    except Exception:
        m = re.search(r"\{.*\}", texto, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return None
    return None


def gerar_par(cliente, modelo, prompt):
    """Pede ao LLM uma pergunta+resposta em JSON. Devolve (pergunta, resposta) ou None."""
    par = extrair_json(_comum.gerar_texto(cliente, modelo, prompt, max_tokens=350, temperature=0.3))
    if par and "pergunta" in par and "resposta" in par:
        return str(par["pergunta"]).strip(), str(par["resposta"]).strip()
    return None


def main():
    parser = argparse.ArgumentParser(description="Gera dataset balanceado (TCU) para a Aula 6.")
    parser.add_argument("--n-docs", type=int, default=80, help="quantos acordaos do TCU amostrar")
    parser.add_argument("--por-tecnica", type=int, default=6, help="perguntas por tecnica")
    parser.add_argument("--seed", type=int, default=42, help="semente da amostragem")
    args = parser.parse_args()

    _comum.carregar_env()
    cliente, modelo = _comum.groq_client()

    print("=" * 60)
    print("  GERAR DATASET (TCU) PARA AS 3 TECNICAS - Aula 6")
    print("=" * 60)

    # 1) Amostra os documentos do TCU e salva o corpus de trabalho.
    todos = _comum.carregar_acordaos_aula4()
    random.seed(args.seed)
    amostra = random.sample(todos, min(args.n_docs, len(todos)))
    corpus_trabalho = [
        {"id": d["id"], "tipo": d.get("tipo", "acordao"),
         "tribunal": d.get("metadata", {}).get("tribunal", "TCU"), "texto": d["texto"]}
        for d in amostra
    ]
    with open(_comum.CORPUS_TRABALHO, "w", encoding="utf-8") as f:
        json.dump(corpus_trabalho, f, ensure_ascii=False, indent=2)
    print(f"Corpus de trabalho: {len(corpus_trabalho)} docs -> {_comum.CORPUS_TRABALHO.name}")

    # 2) Embeddings (p/ clusterizar nas perguntas tematicas do RAPTOR).
    print("Gerando embeddings da amostra (para clusterizar)...")
    docs_hs = _comum.documentos_haystack()  # usa o corpus de trabalho recem salvo
    docs_emb = _comum.doc_embedder().run(documents=docs_hs)["documents"]
    vetores = np.array([d.embedding for d in docs_emb], dtype="float32")
    k = max(2, min(args.por_tecnica, len(corpus_trabalho) // 3))
    rotulos = KMeans(n_clusters=k, n_init=10, random_state=args.seed).fit_predict(vetores)

    perguntas = []

    # 2a) Parent-Child: perguntas especificas (1 doc cada).
    print("Gerando perguntas Parent-Child (especificas)...")
    for i, d in enumerate(amostra[: args.por_tecnica], 1):
        par = gerar_par(cliente, modelo, PROMPT_ESPECIFICA.format(texto=d["texto"][:2500]))
        if par:
            perguntas.append({"id": f"GENPC{i:03d}", "pergunta": par[0], "ground_truth": par[1],
                              "tecnica_ideal": "Parent-Child", "documentos_relevantes": [d["id"]],
                              "tipo": d.get("tipo", "acordao"), "justificativa": JUSTIFICATIVA["Parent-Child"]})

    # 2b) HyDE: perguntas coloquiais (1 doc cada) - usa outra fatia da amostra.
    print("Gerando perguntas HyDE (coloquiais)...")
    fatia_hyde = amostra[args.por_tecnica: args.por_tecnica * 2]
    for i, d in enumerate(fatia_hyde, 1):
        par = gerar_par(cliente, modelo, PROMPT_COLOQUIAL.format(texto=d["texto"][:2500]))
        if par:
            perguntas.append({"id": f"GENHY{i:03d}", "pergunta": par[0], "ground_truth": par[1],
                              "tecnica_ideal": "HyDE", "documentos_relevantes": [d["id"]],
                              "tipo": d.get("tipo", "acordao"), "justificativa": JUSTIFICATIVA["HyDE"]})

    # 2c) RAPTOR: perguntas tematicas (varios docs do mesmo cluster).
    print("Gerando perguntas RAPTOR (tematicas, multi-documento)...")
    for c in range(k):
        if len([1 for r in rotulos if r == c]) < 2:
            continue
        membros = [corpus_trabalho[j] for j in range(len(corpus_trabalho)) if rotulos[j] == c][:3]
        trechos = "\n\n".join(m["texto"][:800] for m in membros)
        par = gerar_par(cliente, modelo, PROMPT_TEMATICA.format(trechos=trechos))
        if par:
            perguntas.append({"id": f"GENRA{c+1:03d}", "pergunta": par[0], "ground_truth": par[1],
                              "tecnica_ideal": "RAPTOR", "documentos_relevantes": [m["id"] for m in membros],
                              "tipo": "tematica", "justificativa": JUSTIFICATIVA["RAPTOR"]})
        if len([p for p in perguntas if p["tecnica_ideal"] == "RAPTOR"]) >= args.por_tecnica:
            break

    with open(_comum.PERGUNTAS_GERADAS, "w", encoding="utf-8") as f:
        json.dump(perguntas, f, ensure_ascii=False, indent=2)

    from collections import Counter
    print(f"\nPerguntas geradas: {len(perguntas)} -> {_comum.PERGUNTAS_GERADAS.name}")
    print("Por tecnica:", dict(Counter(p["tecnica_ideal"] for p in perguntas)))
    print("\nProximo: indexe e compare ->")
    print("  python 02_parent_child.py --recriar")
    print("  python 05_comparar_tecnicas.py")


if __name__ == "__main__":
    main()
