"""
03_compressao_llmlingua.py - Compressao de contexto com LLMLingua-2.

Recupera o contexto (busca densa), COMPRIME com o LLMLingua-2 (remove tokens
previsiveis, preserva o nucleo) e responde com a Groq. Mostra tokens antes/depois e a
economia - util para reduzir custo/latencia em contextos longos.

Precisa: pip install llmlingua  (baixa um modelo encoder multilingue ~560MB; roda em CPU)
e o indice: python 01_indexar.py

Uso:
    python 03_compressao_llmlingua.py --pergunta "o que caracteriza o crime de roubo?"
    python 03_compressao_llmlingua.py --pergunta "..." --taxa 0.4 --top-k 6
"""

import argparse

import _comum

_comum.carregar_env()

MODELO_LLMLINGUA = "microsoft/llmlingua-2-xlm-roberta-large-meetingbank"  # multilingue


def main():
    parser = argparse.ArgumentParser(description="Compressao de contexto (LLMLingua-2).")
    parser.add_argument("--pergunta", required=True)
    parser.add_argument("--taxa", type=float, default=0.5, help="taxa de compressao alvo (0-1)")
    parser.add_argument("--top-k", type=int, default=6)
    args = parser.parse_args()

    store = _comum.abrir_store()
    if store.count_documents() == 0:
        print("[ATENCAO] indice vazio. Rode antes: python 01_indexar.py")
        return

    print("=" * 60)
    print("  COMPRESSAO DE CONTEXTO (LLMLingua-2) - Aula 11")
    print("=" * 60)
    docs = _comum.buscar(_comum.montar_busca(store, args.top_k), args.pergunta)
    contexto = "\n".join(f"- {d.content}" for d in docs)

    try:
        from llmlingua import PromptCompressor
    except ImportError:
        print("[ERRO] instale: pip install llmlingua")
        return

    print(f"Carregando LLMLingua-2 ({MODELO_LLMLINGUA})... (1a vez baixa o modelo)")
    compressor = PromptCompressor(model_name=MODELO_LLMLINGUA, use_llmlingua2=True, device_map="cpu")
    res = compressor.compress_prompt(contexto, rate=args.taxa, force_tokens=["\n", ".", ",", "?"])
    comprimido = res["compressed_prompt"]

    print(f"\nTokens originais : {res.get('origin_tokens')}")
    print(f"Tokens comprimidos: {res.get('compressed_tokens')}")
    print(f"Compressao        : {res.get('ratio')} (taxa alvo {args.taxa})")
    print(f"\n--- contexto comprimido (inicio) ---\n{comprimido[:400]}...\n")

    cliente, modelo = _comum.groq_client()
    resp = _comum.responder_com_contexto(cliente, modelo, args.pergunta, [comprimido])
    print(f"Resposta (com contexto comprimido):\n{resp}")
    print("\nObs.: menos tokens de entrada = menos custo/latencia no LLM grande, "
          "preservando o nucleo semantico.")


if __name__ == "__main__":
    main()
