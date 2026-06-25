"""
03_metricas_retrieval.py - Metricas de qualidade de busca (retrieval).

Mede quao bem o modelo de embedding encontra os documentos certos. Para isso:
  1. Gera embeddings do corpus juridico e de 8 perguntas de avaliacao (Ollama).
  2. Calcula a similaridade entre cada pergunta e cada documento.
  3. Compara com o "gabarito" (qual categoria cada pergunta deveria achar).
  4. Imprime as metricas classicas de RAG:
       - Hit Rate@K : achou pelo menos 1 documento relevante no top-K?
       - Recall@K   : que fracao dos relevantes apareceu no top-K?
       - MRR        : quao no topo veio o primeiro acerto?
       - NDCG@K     : qualidade do ranking (premia acertos no topo)

Precisa do Ollama rodando. Nao precisa de OpenSearch nem Groq.

Uso:
    python 03_metricas_retrieval.py
    python 03_metricas_retrieval.py --modelo nomic-embed-text --k 1 3 5
"""

import argparse

import numpy as np
import requests
from sklearn.metrics import ndcg_score
from sklearn.metrics.pairwise import cosine_similarity

import _comum

# Perguntas de avaliacao + a categoria que cada uma deveria recuperar (gabarito).
QUERIES = [
    {"query": "desvio de dinheiro publico por funcionario do governo", "categoria_alvo": "Crimes Funcionais"},
    {"query": "pedido de soltura por falta de fundamentacao", "categoria_alvo": "Direito Processual"},
    {"query": "arrombamento de casa para roubo de bens", "categoria_alvo": "Crimes Patrimoniais"},
    {"query": "exame pericial em local de homicidio com analise balistica", "categoria_alvo": "Crimes Contra a Vida"},
    {"query": "investigacao federal sobre organizacao criminosa de trafico", "categoria_alvo": "Investigação Criminal"},
    {"query": "agressao fisica contra mulher em ambiente familiar", "categoria_alvo": "Violência Doméstica"},
    {"query": "golpe aplicado em idoso com promessa de premio", "categoria_alvo": "Crimes Patrimoniais"},
    {"query": "medida cautelar com tornozeleira eletronica", "categoria_alvo": "Direito Processual"},
]

# Categorias "proximas" recebem relevancia parcial (grau 1) no NDCG.
AFINIDADE = {
    "Crimes Funcionais": ["Investigação Criminal"],
    "Investigação Criminal": ["Crimes Funcionais", "Direito Processual"],
    "Direito Processual": ["Investigação Criminal"],
    "Crimes Patrimoniais": [],
    "Crimes Contra a Vida": ["Violência Doméstica"],
    "Violência Doméstica": ["Crimes Contra a Vida"],
}


def gerar_embeddings(textos, base_url, modelo):
    vetores = []
    for t in textos:
        r = requests.post(f"{base_url}/api/embeddings",
                          json={"model": modelo, "prompt": t}, timeout=60)
        r.raise_for_status()
        vetores.append(r.json()["embedding"])
    return np.array(vetores, dtype="float32")


def grau_relevancia(categoria_doc, query):
    """2 = mesma categoria, 1 = categoria proxima, 0 = irrelevante."""
    if categoria_doc == query["categoria_alvo"]:
        return 2
    if categoria_doc in AFINIDADE.get(query["categoria_alvo"], []):
        return 1
    return 0


def hit_rate_at_k(sims, relev, k):
    acertos = 0
    for i in range(sims.shape[0]):
        topk = np.argsort(sims[i])[::-1][:k]
        if (relev[i, topk] >= 1).any():
            acertos += 1
    return acertos / sims.shape[0]


def recall_at_k(sims, relev, k):
    recalls = []
    for i in range(sims.shape[0]):
        total_rel = (relev[i] >= 1).sum()
        if total_rel == 0:
            continue
        topk = np.argsort(sims[i])[::-1][:k]
        recalls.append((relev[i, topk] >= 1).sum() / total_rel)
    return float(np.mean(recalls))


def mrr(sims, relev):
    rrs = []
    for i in range(sims.shape[0]):
        ranking = np.argsort(sims[i])[::-1]
        rrs.append(0.0)
        for posicao, idx in enumerate(ranking, start=1):
            if relev[i, idx] >= 1:
                rrs[-1] = 1.0 / posicao
                break
    return float(np.mean(rrs))


def main():
    parser = argparse.ArgumentParser(description="Metricas de retrieval para RAG.")
    parser.add_argument("--modelo", default=None, help="modelo de embedding (padrao: .env)")
    parser.add_argument("--k", type=int, nargs="+", default=[1, 3, 5],
                        help="valores de K para Hit Rate e Recall (padrao: 1 3 5)")
    args = parser.parse_args()

    _comum.carregar_env()
    base_url, modelo_padrao = _comum.config_ollama()
    modelo = args.modelo or modelo_padrao
    corpus = _comum.carregar_corpus()

    print("=" * 60)
    print(f"  METRICAS DE RETRIEVAL - Aula 1 (modelo: {modelo})")
    print("=" * 60)
    print(f"Corpus: {len(corpus)} documentos | Perguntas: {len(QUERIES)}")

    # Monta a matriz de relevancia (gabarito): linhas = perguntas, colunas = docs.
    relev = np.zeros((len(QUERIES), len(corpus)), dtype=int)
    for i, q in enumerate(QUERIES):
        for j, doc in enumerate(corpus):
            relev[i, j] = grau_relevancia(doc["categoria"], q)

    print("\nGerando embeddings (corpus + perguntas) via Ollama...")
    emb_corpus = gerar_embeddings([d["texto"] for d in corpus], base_url, modelo)
    emb_queries = gerar_embeddings([q["query"] for q in QUERIES], base_url, modelo)
    sims = cosine_similarity(emb_queries, emb_corpus)

    print("\nRESULTADO")
    print("-" * 40)
    for k in args.k:
        print(f"  Hit Rate@{k}: {hit_rate_at_k(sims, relev, k):.3f}")
    for k in args.k:
        print(f"  Recall@{k}:   {recall_at_k(sims, relev, k):.3f}")
    print(f"  MRR:         {mrr(sims, relev):.3f}")
    print(f"  NDCG@5:      {ndcg_score(relev, sims, k=5):.3f}")
    print(f"  NDCG@10:     {ndcg_score(relev, sims, k=10):.3f}")
    print("\nDica: quanto mais perto de 1.0, melhor o modelo encontra os documentos certos.")


if __name__ == "__main__":
    main()
