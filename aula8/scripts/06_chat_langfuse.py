"""
06_chat_langfuse.py - Chat CRAG interativo instrumentado no LangFuse.

Usa o MESMO pipeline CRAG do 04 (retrieve -> avaliar -> ConditionalRouter ->
busca_web -> montar -> gerar). Como tudo esta num pipeline Haystack, a
auto-instrumentacao do LangFuse captura cada etapa no trace 'crag-aula8' - inclusive
o avaliador, a rota tomada e (quando ocorre) o web search.

A cada pergunta o chat mostra a ROTA escolhida e o link do trace, para voce comparar
no LangFuse as perguntas que ficam locais vs. as que disparam o fallback web.

Uso:
    python 06_chat_langfuse.py
    python 06_chat_langfuse.py --top-k 4 --indice aula4_hibrido
"""

import argparse

import _comum

_comum.carregar_env()


def main():
    parser = argparse.ArgumentParser(description="Chat CRAG + LangFuse (Aula 8).")
    parser.add_argument("--indice", default=_comum.INDICE_TCU)
    parser.add_argument("--top-k", type=int, default=4)
    args = parser.parse_args()

    usar_langfuse = _comum.langfuse_configurado()

    print("=" * 60)
    print("  CHAT CRAG + LANGFUSE - Aula 8")
    print("=" * 60)
    print(f"Indice: {args.indice}")
    print(f"LangFuse: {'ligado (trace crag-aula8)' if usar_langfuse else 'DESLIGADO (sem chaves)'}")
    print(f"Tavily: {'real' if _comum.tavily_configurado() else 'OFFLINE (stub)'}")

    store = _comum.abrir_store(args.indice)
    try:
        if store.count_documents() == 0:
            print("\n[ATENCAO] Indice vazio. Rode antes: python 01_indexar_opensearch.py")
            return
    except Exception as e:
        print(f"[ATENCAO] nao consegui acessar o indice: {e}")
        return

    modulo_crag = _comum.importar_script("04_crag.py")
    pipe = modulo_crag.montar_pipeline(store, args.top_k, usar_langfuse)

    print("\nDigite sua pergunta. Para sair: 'sair' (ou Ctrl+C).\n")
    while True:
        try:
            pergunta = input("Pergunta> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nAte logo!")
            break
        if pergunta.lower() in {"sair", "exit", "quit"} or not pergunta:
            print("Ate logo!")
            break
        try:
            resultado = modulo_crag.responder(pipe, pergunta, usar_langfuse)
        except Exception as e:
            print(f"  [erro] {e}\n")
            continue
        score = resultado["avaliar"]["score"]
        rota = resultado["avaliar"]["rota"]
        docs = resultado["montar"]["documents"]
        resposta = resultado["llm"]["replies"][0]
        print(f"\n  Score local: {score:.2f} | ROTA: {rota.upper()}")
        print(f"  Fontes: {[d.meta.get('id_original') for d in docs]}")
        print(f"  Resposta: {resposta[:500]}")
        if usar_langfuse:
            url = resultado.get("tracer", {}).get("trace_url", "")
            if url:
                print(f"  Trace: {url}")
        print()


if __name__ == "__main__":
    main()
