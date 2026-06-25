"""
03_step_back.py - Step-Back Prompting (#T11).

Problema: perguntas muito especificas as vezes "erram" a busca porque o detalhe
nao casa com nenhum documento. Step-Back gera uma pergunta MAIS GERAL (sobe um
nivel de abstracao) e busca pelos dois (a especifica + a geral), juntando os
resultados. A geral costuma trazer o documento que contem o detalhe.

Fluxo: pergunta -> LLM gera pergunta geral -> busca (especifica + geral) -> dedup -> resposta.

Precisa de OpenSearch (indice do TCU), Ollama e Groq.

Uso:
    python 03_step_back.py
    python 03_step_back.py --pergunta "qual o prazo para recolher a multa do acordao X?"
"""

import argparse

import _comum

QUERY_EXEMPLO = "qual foi o valor do debito imputado no acordao sobre o convenio de pavimentacao?"


def construir_responder(indice=None, top_k=5):
    """Devolve responder(pergunta) -> dict (Step-Back Prompting)."""
    indice = indice or _comum.INDICE_TCU
    store = _comum.abrir_store(indice)
    pipe = _comum.montar_busca(store, top_k)
    cliente, modelo = _comum.groq_client()

    def responder(pergunta):
        geral = _comum.gerar_stepback(cliente, modelo, pergunta)
        listas = [_comum.buscar(pipe, pergunta), _comum.buscar(pipe, geral)]
        docs = _comum.dedup_por_id(listas, top_k)
        contextos = [d.content for d in docs]
        resposta = _comum.responder_com_contexto(cliente, modelo, pergunta, contextos)
        return {"resposta": resposta, "contextos": contextos,
                "ids": [d.meta.get("id_original") for d in docs],
                "pergunta_geral": geral, "n_buscas": 2, "n_llm": 2}

    return responder


def main():
    parser = argparse.ArgumentParser(description="Step-Back Prompting (#T11).")
    parser.add_argument("--pergunta", default=QUERY_EXEMPLO)
    parser.add_argument("--indice", default=_comum.INDICE_TCU)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    _comum.carregar_env()
    print("=" * 60)
    print("  STEP-BACK PROMPTING (#T11) - Aula 7")
    print("=" * 60)
    print(f"Pergunta: {args.pergunta}")
    responder = construir_responder(args.indice, args.top_k)
    r = responder(args.pergunta)
    print(f"\nPergunta geral (step-back): {r['pergunta_geral']}")
    print(f"Documentos recuperados ({len(r['ids'])}): {r['ids']}")
    print(f"Custo: {r['n_buscas']} buscas + {r['n_llm']} chamadas LLM")
    print(f"\nResposta:\n{r['resposta']}")


if __name__ == "__main__":
    main()
