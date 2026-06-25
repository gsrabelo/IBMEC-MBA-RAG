"""
02_multi_query.py - Multi-Query RAG (#T10).

Problema (vocabulary mismatch): a pergunta do usuario pode usar palavras diferentes
dos documentos. Multi-Query gera VARIAS reformulacoes da pergunta (com vocabulario
diferente), busca por TODAS e junta os resultados (removendo duplicados). Assim
aumenta a chance de achar o documento certo.

Fluxo: pergunta -> LLM gera N variacoes -> busca cada uma -> dedup por id -> resposta.

DOIS MODOS (para comparar didaticamente):
  - manual (padrao): nos mesmos buscamos cada variacao e deduplicamos com
    _comum.dedup_por_id (por id, mantendo o de maior score).
  - --nativo: usa o componente oficial do Haystack MultiQueryEmbeddingRetriever, que
    faz o retrieval EM PARALELO e deduplica internamente (_deduplicate_documents,
    mesma regra: por id, maior score). Em AMBOS, as variacoes vem do LLM (Groq) -
    o componente nativo NAO gera variacoes, so recebe a lista pronta.

Precisa de OpenSearch (indice do TCU), Ollama e Groq.

Uso:
    python 02_multi_query.py
    python 02_multi_query.py --pergunta "o orgao pode contratar sem licitacao?" --n 4
    python 02_multi_query.py --nativo        # usa o MultiQueryEmbeddingRetriever
"""

import argparse

import _comum

QUERY_EXEMPLO = "quando o gestor pode ser multado pelo tribunal de contas?"


def _criar_multiquery_nativo(store, top_k):
    """Monta o MultiQueryEmbeddingRetriever oficial do Haystack (retrieval paralelo)."""
    from haystack.components.retrievers import MultiQueryEmbeddingRetriever
    from haystack_integrations.components.retrievers.opensearch import (
        OpenSearchEmbeddingRetriever,
    )

    retriever = OpenSearchEmbeddingRetriever(document_store=store, top_k=top_k)
    mqr = MultiQueryEmbeddingRetriever(
        retriever=retriever, query_embedder=_comum.text_embedder(), max_workers=4)
    mqr.warm_up()
    return mqr


def construir_responder(indice=None, top_k=5, n_variacoes=4, nativo=False):
    """Devolve responder(pergunta) -> dict (Multi-Query RAG), manual ou nativo."""
    indice = indice or _comum.INDICE_TCU
    store = _comum.abrir_store(indice)
    cliente, modelo = _comum.groq_client()

    if nativo:
        mqr = _criar_multiquery_nativo(store, top_k)

        def responder(pergunta):
            variacoes = _comum.gerar_variacoes(cliente, modelo, pergunta, n=n_variacoes)
            consultas = [pergunta] + variacoes
            # o componente nativo embeda, busca em paralelo e ja deduplica + ordena
            docs = mqr.run(queries=consultas)["documents"][:top_k]
            contextos = [d.content for d in docs]
            resposta = _comum.responder_com_contexto(cliente, modelo, pergunta, contextos)
            return {"resposta": resposta, "contextos": contextos,
                    "ids": [d.meta.get("id_original") for d in docs],
                    "variacoes": variacoes, "n_buscas": len(consultas), "n_llm": 2,
                    "modo": "nativo (MultiQueryEmbeddingRetriever, retrieval paralelo)"}
    else:
        pipe = _comum.montar_busca(store, top_k)

        def responder(pergunta):
            variacoes = _comum.gerar_variacoes(cliente, modelo, pergunta, n=n_variacoes)
            consultas = [pergunta] + variacoes
            listas = [_comum.buscar(pipe, c) for c in consultas]
            docs = _comum.dedup_por_id(listas, top_k)
            contextos = [d.content for d in docs]
            resposta = _comum.responder_com_contexto(cliente, modelo, pergunta, contextos)
            return {"resposta": resposta, "contextos": contextos,
                    "ids": [d.meta.get("id_original") for d in docs],
                    "variacoes": variacoes, "n_buscas": len(consultas), "n_llm": 2,
                    "modo": "manual (dedup_por_id, retrieval sequencial)"}

    return responder


def main():
    parser = argparse.ArgumentParser(description="Multi-Query RAG (#T10).")
    parser.add_argument("--pergunta", default=QUERY_EXEMPLO)
    parser.add_argument("--indice", default=_comum.INDICE_TCU)
    parser.add_argument("--n", type=int, default=4, help="numero de variacoes")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--nativo", action="store_true",
                        help="usa o MultiQueryEmbeddingRetriever do Haystack (em vez do manual)")
    args = parser.parse_args()

    _comum.carregar_env()
    print("=" * 60)
    print("  MULTI-QUERY RAG (#T10) - Aula 7")
    print("=" * 60)
    responder = construir_responder(args.indice, args.top_k, args.n, nativo=args.nativo)
    r = responder(args.pergunta)
    print(f"Modo: {r['modo']}")
    print(f"Pergunta: {args.pergunta}")
    print("\nVariacoes geradas (LLM):")
    for i, v in enumerate(r["variacoes"], 1):
        print(f"  {i}. {v}")
    print(f"\nDocumentos recuperados ({len(r['ids'])}): {r['ids']}")
    print(f"Custo: {r['n_buscas']} buscas + {r['n_llm']} chamadas LLM")
    print(f"\nResposta:\n{r['resposta']}")


if __name__ == "__main__":
    main()
