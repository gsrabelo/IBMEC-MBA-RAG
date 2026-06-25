"""
03_raptor.py - RAPTOR (#T08) - versao compacta com Haystack.

RAPTOR cria uma "arvore de resumos": agrupa documentos parecidos em clusters e
gera um RESUMO de cada cluster. A busca passa a ter dois niveis:
  - documentos ORIGINAIS  -> detalhes especificos
  - RESUMOS dos clusters  -> visao geral (perguntas amplas/tematicas)
Indexamos os dois juntos (estrategia "collapsed tree").

Rastreabilidade: cada resumo guarda os IDs dos documentos de origem
(meta['documentos_origem']) e cada original guarda seu cluster (meta['cluster']),
para voce conseguir ligar resumo <-> documentos originais.

Obs.: o pack oficial llama-index-packs-raptor esta descontinuado; aqui fazemos
uma versao didatica equivalente no stack do curso (Haystack + Groq + Ollama).

Precisa de OpenSearch, Ollama e Groq.

Uso:
    python 03_raptor.py --recriar
    python 03_raptor.py --pergunta "qual o tema geral dos acordaos sobre prisao?"
"""

import argparse

import numpy as np
import requests
from haystack import Document, Pipeline
from haystack.document_stores.types import DuplicatePolicy
from haystack_integrations.components.retrievers.opensearch import (
    OpenSearchEmbeddingRetriever,
)

import _comum

PROMPT_RESUMO = (
    "Resuma em um paragrafo os trechos juridicos abaixo, preservando os pontos "
    "principais (orgao/tribunal, tema e o que se decide). Seja conciso.\n\n{trechos}"
)


def agrupar(vetores):
    """Reduz dimensoes (UMAP) e agrupa em clusters (GMM, melhor k por BIC).

    Devolve um array de rotulos (1 por documento). Em caso de falha, 1 cluster so.
    """
    n = len(vetores)
    if n < 4:
        return np.zeros(n, dtype=int)
    try:
        import umap
        from sklearn.mixture import GaussianMixture

        n_comp = min(5, n - 2)
        reduzido = umap.UMAP(n_neighbors=min(10, n - 1), n_components=n_comp,
                             random_state=42).fit_transform(vetores)
        melhor_k, melhor_bic, melhor_modelo = 1, np.inf, None
        for k in range(2, min(6, n)):
            gmm = GaussianMixture(n_components=k, covariance_type="diag", random_state=42)
            gmm.fit(reduzido)
            bic = gmm.bic(reduzido)
            if bic < melhor_bic:
                melhor_k, melhor_bic, melhor_modelo = k, bic, gmm
        return melhor_modelo.predict(reduzido) if melhor_modelo else np.zeros(n, dtype=int)
    except Exception as e:
        print(f"  (aviso: clustering falhou, usando 1 cluster: {e})")
        return np.zeros(n, dtype=int)


def construir_responder(indice, top_k=5, recriar=False):
    """Monta o indice RAPTOR (originais + resumos) e devolve responder(pergunta)."""
    os_cfg = _comum.config_opensearch()
    auth = (os_cfg["usuario"], os_cfg["senha"]) if os_cfg["usuario"] else None
    if recriar:
        try:
            requests.delete(f"{os_cfg['url']}/{indice}", auth=auth, timeout=10)
        except Exception:
            pass

    docs = _comum.documentos_haystack()
    embedder = _comum.doc_embedder()
    docs_emb = embedder.run(documents=docs)["documents"]
    vetores = np.array([d.embedding for d in docs_emb], dtype="float32")

    # Agrupa e resume cada cluster com a Groq.
    rotulos = agrupar(vetores)
    cliente, modelo = _comum.groq_client()
    resumos = []
    for cluster in sorted(set(rotulos.tolist())):
        membros = [docs[i] for i in range(len(docs)) if rotulos[i] == cluster]
        trechos = "\n\n".join(d.content[:800] for d in membros)
        try:
            texto = _comum.gerar_texto(cliente, modelo, PROMPT_RESUMO.format(trechos=trechos), max_tokens=350)
        except Exception as e:
            texto = ""
            print(f"  (aviso: resumo do cluster {cluster} falhou: {e})")
        if texto:
            print(texto)
            ids_origem = [m.meta.get("id_original") for m in membros]
            resumos.append(Document(content=texto,
                                    meta={"tipo": "resumo_raptor", "cluster": int(cluster),
                                          "n_membros": len(membros), "documentos_origem": ids_origem}))
    print(f"  Clusters: {len(set(rotulos.tolist()))} | resumos gerados: {len(resumos)}")

    # Rotula cada documento original com o cluster a que pertence (rastreabilidade).
    for i, d in enumerate(docs_emb):
        d.meta["cluster"] = int(rotulos[i])

    # Indexa originais + resumos (todos com embedding) no OpenSearch.
    store = _comum.abrir_store(indice)
    resumos_emb = embedder.run(documents=resumos)["documents"] if resumos else []
    store.write_documents(docs_emb + resumos_emb, policy=DuplicatePolicy.OVERWRITE)

    busca = Pipeline()
    busca.add_component("embedder", _comum.text_embedder())
    busca.add_component("retriever", OpenSearchEmbeddingRetriever(document_store=store, top_k=top_k))
    busca.connect("embedder.embedding", "retriever.query_embedding")

    def descrever(d):
        """Descreve a origem de um trecho recuperado (resumo de cluster ou doc original)."""
        if d.meta.get("tipo") == "resumo_raptor":
            return {"tipo": "resumo", "cluster": d.meta.get("cluster"),
                    "documentos_origem": d.meta.get("documentos_origem", [])}
        return {"tipo": "original", "id": d.meta.get("id_original"),
                "cluster": d.meta.get("cluster")}

    def responder(pergunta):
        docs_r = busca.run({"embedder": {"text": pergunta}})["retriever"]["documents"]
        contextos = [d.content for d in docs_r]
        fontes = [descrever(d) for d in docs_r]
        resposta = _comum.responder_com_contexto(cliente, modelo, pergunta, contextos)
        return resposta, contextos, fontes

    return responder


def main():
    parser = argparse.ArgumentParser(description="RAPTOR compacto (#T08).")
    parser.add_argument("--indice", default="aula6_raptor", help="indice no OpenSearch")
    parser.add_argument("--pergunta", default=None, help="pergunta (senao roda exemplos do corpus)")
    parser.add_argument("--top-k", type=int, default=5, help="quantos trechos buscar")
    parser.add_argument("--recriar", action="store_true", help="recria o indice")
    args = parser.parse_args()

    _comum.carregar_env()
    print("=" * 60)
    print("  RAPTOR (#T08) - Aula 6")
    print("=" * 60)
    print(f"Indice: {args.indice} | top_k={args.top_k}")
    print("Construindo a arvore (embeddings -> clusters -> resumos)...")

    responder = construir_responder(args.indice, args.top_k, args.recriar)

    _, perguntas = _comum.carregar_corpus()
    alvos = [args.pergunta] if args.pergunta else [p["pergunta"] for p in perguntas][:3]
    for q in alvos:
        print("\n" + "-" * 55)
        print(f"Pergunta: {q}")
        resposta, contextos, fontes = responder(q)
        print(f"Trechos usados: {len(contextos)}")
        print("Fontes (origem de cada trecho):")
        for i, fo in enumerate(fontes, 1):
            if fo["tipo"] == "resumo":
                print(f"  {i}. RESUMO do cluster {fo['cluster']} <- docs {fo['documentos_origem']}")
            else:
                print(f"  {i}. ORIGINAL {fo['id']} (cluster {fo['cluster']})")
        print(f"Resposta: {resposta}")


if __name__ == "__main__":
    main()
