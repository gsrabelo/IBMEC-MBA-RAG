"""
04_rag_fusion.py - RAG-Fusion (#T12).

RAG-Fusion = Multi-Query + fusao por RRF. Gera varias variacoes da pergunta, busca
por todas e FUNDE os rankings com Reciprocal Rank Fusion (RRF): cada documento
ganha pontos por 1/(k + posicao) em cada lista. Documentos que aparecem bem em
VARIAS variacoes sobem ao topo - resultado mais robusto que a deduplicacao simples.

Fluxo: pergunta -> LLM gera N variacoes -> busca cada uma -> RRF -> resposta.

Precisa de OpenSearch (indice do TCU), Ollama e Groq.

Uso:
    python 04_rag_fusion.py
    python 04_rag_fusion.py --pergunta "o orgao pode aderir a ata de outro?" --n 4
"""

import argparse

import _comum

QUERY_EXEMPLO = "quando o tribunal de contas considera as contas irregulares?"


def construir_responder(indice=None, top_k=5, n_variacoes=4):
    """Devolve responder(pergunta) -> dict (RAG-Fusion)."""
    indice = indice or _comum.INDICE_TCU
    store = _comum.abrir_store(indice)
    pipe = _comum.montar_busca(store, top_k)
    cliente, modelo = _comum.groq_client()

    def responder(pergunta):
        variacoes = _comum.gerar_variacoes(cliente, modelo, pergunta, n=n_variacoes)
        consultas = [pergunta] + variacoes
        listas = [_comum.buscar(pipe, c) for c in consultas]
        docs = _comum.fundir_rrf(listas, top_k)
        contextos = [d.content for d in docs]
        resposta = _comum.responder_com_contexto(cliente, modelo, pergunta, contextos)
        return {"resposta": resposta, "contextos": contextos,
                "ids": [d.meta.get("id_original") for d in docs],
                "variacoes": variacoes, "n_buscas": len(consultas), "n_llm": 2}

    return responder


def main():
    parser = argparse.ArgumentParser(description="RAG-Fusion (#T12).")
    parser.add_argument("--pergunta", default=QUERY_EXEMPLO)
    parser.add_argument("--indice", default=_comum.INDICE_TCU)
    parser.add_argument("--n", type=int, default=4, help="numero de variacoes")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    _comum.carregar_env()
    print("=" * 60)
    print("  RAG-FUSION (#T12) - Aula 7")
    print("=" * 60)
    print(f"Pergunta: {args.pergunta}")
    responder = construir_responder(args.indice, args.top_k, args.n)
    r = responder(args.pergunta)
    print("\nVariacoes geradas:")
    for i, v in enumerate(r["variacoes"], 1):
        print(f"  {i}. {v}")
    print(f"\nDocumentos recuperados ({len(r['ids'])}, ordenados por RRF): {r['ids']}")
    print(f"Custo: {r['n_buscas']} buscas + {r['n_llm']} chamadas LLM")
    print(f"\nResposta:\n{r['resposta']}")


if __name__ == "__main__":
    main()
