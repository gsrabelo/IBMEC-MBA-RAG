"""
00_check_ambiente.py - Confere se o ambiente da Aula 8 esta pronto.

Verifica: OpenSearch (indice TCU), Ollama (embeddings), Groq (LLM/avaliador),
LangFuse (observabilidade, opcional) e Tavily (web search de fallback, opcional).

Uso:
    python 00_check_ambiente.py
    python 00_check_ambiente.py --testar-groq      # faz 1 chamada real na Groq
"""

import argparse

import _comum

_comum.carregar_env()


def ok(b):
    return "OK" if b else "FALHOU"


def checar_opensearch():
    try:
        store = _comum.abrir_store(_comum.INDICE_TCU)
        n = store.count_documents()
        print(f"[OpenSearch] {ok(True)} - indice '{_comum.INDICE_TCU}' com {n} documentos")
        if n == 0:
            print("             (vazio) rode: python 01_indexar_opensearch.py")
        return True
    except Exception as e:
        print(f"[OpenSearch] {ok(False)} - {e}")
        return False


def checar_ollama():
    try:
        emb = _comum.text_embedder()
        if hasattr(emb, "warm_up"):
            emb.warm_up()
        v = emb.run(text="teste de embedding")["embedding"]
        print(f"[Ollama]     {ok(True)} - embedding com {len(v)} dimensoes")
        return True
    except Exception as e:
        print(f"[Ollama]     {ok(False)} - {e}")
        return False


def checar_groq(testar):
    api_key, modelo, _ = _comum.config_groq()
    if not api_key:
        print(f"[Groq]       {ok(False)} - GROQ_API_KEY ausente no .env")
        return False
    if not testar:
        print(f"[Groq]       {ok(True)} - chave presente (modelo {modelo}); use --testar-groq p/ chamada real")
        return True
    try:
        cliente, modelo = _comum.groq_client()
        r = _comum.gerar_texto(cliente, modelo, "Responda apenas: ok", max_tokens=5)
        print(f"[Groq]       {ok(True)} - resposta: {r!r}")
        return True
    except Exception as e:
        print(f"[Groq]       {ok(False)} - {e}")
        return False


def checar_langfuse():
    if _comum.langfuse_configurado():
        print(f"[LangFuse]   {ok(True)} - chaves presentes (tracing ligado no 06)")
    else:
        print("[LangFuse]   (opcional) sem chaves - o 06 roda sem tracing")
    return True


def checar_tavily():
    try:
        from haystack_integrations.components.websearch.tavily import TavilyWebSearch  # noqa: F401
        tem_pacote = True
    except Exception:
        tem_pacote = False

    if _comum.tavily_configurado() and tem_pacote:
        print("[Tavily]     OK - chave + tavily-haystack presentes (web search real)")
    elif _comum.tavily_configurado() and not tem_pacote:
        print("[Tavily]     (parcial) chave presente, falta 'pip install tavily-haystack' -> fallback OFFLINE")
    elif tem_pacote:
        print("[Tavily]     (opcional) pacote presente, sem TAVILY_API_KEY -> fallback OFFLINE")
    else:
        print("[Tavily]     (opcional) sem chave e sem pacote -> CRAG usa fallback OFFLINE (stub)")
    return True


def main():
    parser = argparse.ArgumentParser(description="Checagem de ambiente da Aula 8.")
    parser.add_argument("--testar-groq", action="store_true", help="faz 1 chamada real na Groq")
    args = parser.parse_args()

    print("=" * 60)
    print("  CHECAGEM DE AMBIENTE - Aula 8 (Self-RAG / CRAG)")
    print("=" * 60)
    resultados = [
        checar_opensearch(),
        checar_ollama(),
        checar_groq(args.testar_groq),
        checar_langfuse(),
        checar_tavily(),
    ]
    print("-" * 60)
    obrigatorios = resultados[:3]  # OpenSearch, Ollama, Groq
    if all(obrigatorios):
        print("Tudo pronto para a Aula 8.")
    else:
        print("Resolva os itens FALHOU (obrigatorios) antes de seguir.")


if __name__ == "__main__":
    main()
