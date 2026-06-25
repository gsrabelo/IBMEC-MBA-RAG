"""
02_embeddings_similaridade.py - Embeddings e similaridade de cosseno.

Gera vetores (embeddings) para os documentos do corpus juridico
(datasets/corpus_juridico_aula1.json) usando o Ollama e mostra quais documentos
sao parecidos entre si (similaridade de cosseno).

Opcional: com --umap, projeta os vetores em 2D e salva um grafico PNG mostrando
os "grupos" de documentos parecidos (coloridos por categoria).

Uso:
    python 02_embeddings_similaridade.py
    python 02_embeddings_similaridade.py --modelo nomic-embed-text
    python 02_embeddings_similaridade.py --umap
    python 02_embeddings_similaridade.py --n 10
    python 02_embeddings_similaridade.py --frase1 "prisao preventiva" --frase2 "soltura do reu"
"""

import argparse

import numpy as np
import requests
from sklearn.metrics.pairwise import cosine_similarity

import _comum


def gerar_embedding(texto, base_url, modelo):
    """Pede ao Ollama o vetor (embedding) de um texto."""
    resposta = requests.post(
        f"{base_url}/api/embeddings",
        json={"model": modelo, "prompt": texto},
        timeout=60,
    )
    resposta.raise_for_status()
    return resposta.json()["embedding"]


def gerar_embeddings(textos, base_url, modelo):
    """Gera os vetores de uma lista de textos e devolve um array numpy."""
    vetores = [gerar_embedding(t, base_url, modelo) for t in textos]
    return np.array(vetores, dtype="float32")


def mostrar_similaridade(rotulos, vetores):
    print("\nSIMILARIDADE ENTRE OS DOCUMENTOS (1.0 = identicos, 0 = sem relacao)")
    print("-" * 60)
    sim = cosine_similarity(vetores)
    # Para cada documento, mostra o documento MAIS parecido (fora ele mesmo).
    for i, rotulo in enumerate(rotulos):
        linha = sim[i].copy()
        linha[i] = -1  # ignora a comparacao consigo mesmo
        j = int(np.argmax(linha))
        print(f"[{i+1}] {rotulo[:50]:50s}")
        print(f"     mais parecido -> [{j+1}] {rotulos[j][:42]} (score {sim[i][j]:.3f})")


def comparar_par(frase1, frase2, base_url, modelo):
    print("\nCOMPARACAO DIRETA DE DUAS FRASES")
    print("-" * 60)
    vetores = gerar_embeddings([frase1, frase2], base_url, modelo)
    score = cosine_similarity([vetores[0]], [vetores[1]])[0][0]
    print(f"  Frase 1: {frase1}")
    print(f"  Frase 2: {frase2}")
    print(f"  Similaridade de cosseno: {score:.4f}")


def salvar_umap(rotulos, categorias, vetores, arquivo):
    """Projeta os vetores em 2D com UMAP e salva um PNG."""
    import matplotlib.pyplot as plt
    import umap

    print("\nGerando mapa UMAP em 2D...")
    reducer = umap.UMAP(n_neighbors=4, min_dist=0.3, random_state=42)
    pontos = reducer.fit_transform(vetores)

    plt.figure(figsize=(9, 7))
    cats = sorted(set(categorias))
    cores = plt.cm.tab10(np.linspace(0, 1, len(cats)))
    for cat, cor in zip(cats, cores):
        idx = [k for k, c in enumerate(categorias) if c == cat]
        plt.scatter(pontos[idx, 0], pontos[idx, 1], color=cor, label=cat, s=120)
    for k in range(len(rotulos)):
        plt.annotate(str(k + 1), (pontos[k, 0], pontos[k, 1]),
                     fontsize=11, fontweight="bold")
    plt.title("Mapa semantico (UMAP) dos documentos juridicos")
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(arquivo, dpi=120)
    print(f"Grafico salvo em: {arquivo}")
    print("(os numeros no grafico correspondem aos documentos listados acima)")


def main():
    parser = argparse.ArgumentParser(description="Embeddings e similaridade de cosseno.")
    parser.add_argument("--modelo", default=None,
                        help="modelo de embedding do Ollama (padrao: o do .env)")
    parser.add_argument("--n", type=int, default=8,
                        help="quantos documentos do corpus usar (padrao: 8)")
    parser.add_argument("--umap", action="store_true",
                        help="gera e salva um mapa 2D (UMAP) dos documentos")
    parser.add_argument("--frase1", help="primeira frase para comparacao direta")
    parser.add_argument("--frase2", help="segunda frase para comparacao direta")
    args = parser.parse_args()

    _comum.carregar_env()
    base_url, modelo_padrao = _comum.config_ollama()
    modelo = args.modelo or modelo_padrao

    print("=" * 60)
    print(f"  EMBEDDINGS & SIMILARIDADE - Aula 1 (modelo: {modelo})")
    print("=" * 60)

    # Modo 1: comparar duas frases informadas pelo usuario.
    if args.frase1 and args.frase2:
        comparar_par(args.frase1, args.frase2, base_url, modelo)
        return

    # Modo 2 (padrao): usar os documentos do corpus juridico.
    corpus = _comum.carregar_corpus()[: args.n]
    rotulos = [d["titulo"] for d in corpus]
    categorias = [d["categoria"] for d in corpus]
    textos = [d["texto"] for d in corpus]

    print(f"\nGerando embeddings de {len(corpus)} documentos do corpus via Ollama...")
    vetores = gerar_embeddings(textos, base_url, modelo)
    print(f"Pronto. Cada documento virou um vetor de {vetores.shape[1]} numeros.")

    mostrar_similaridade(rotulos, vetores)

    if args.umap:
        salvar_umap(rotulos, categorias, vetores, _comum.PASTA_SCRIPTS / "umap_documentos.png")


if __name__ == "__main__":
    main()
