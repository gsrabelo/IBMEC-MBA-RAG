"""
00_check_ambiente.py - Confere se o ambiente da Aula 9 (Graph RAG / LightRAG) esta pronto.

Verifica: pacote lightrag-hku, Groq (LLM), Ollama (embeddings) e o corpus.
Nesta aula NAO usamos OpenSearch nem LangFuse (storage do LightRAG e em arquivo).

Uso:
    python 00_check_ambiente.py
    python 00_check_ambiente.py --testar-groq   # faz 1 chamada real na Groq
"""

import argparse

import _comum

_comum.carregar_env()


def ok(b):
    return "OK" if b else "FALHOU"


def checar_lightrag():
    try:
        import lightrag  # noqa: F401
        from lightrag import LightRAG, QueryParam  # noqa: F401
        print(f"[LightRAG]   {ok(True)} - pacote lightrag-hku instalado")
        return True
    except Exception as e:
        print(f"[LightRAG]   {ok(False)} - instale: pip install lightrag-hku  ({e})")
        return False


def checar_corpus():
    if _comum.CORPUS.exists():
        n = len(_comum.ler_corpus())
        print(f"[Corpus]     {ok(True)} - {_comum.CORPUS.name} ({n} caracteres)")
        return True
    print(f"[Corpus]     {ok(False)} - nao encontrado: {_comum.CORPUS}")
    return False


def checar_ollama():
    try:
        from lightrag.llm.ollama import ollama_embed

        base_url, modelo = _comum.config_ollama()
        import asyncio

        v = asyncio.run(ollama_embed.func(["teste de embedding"], embed_model=modelo, host=base_url))
        dim = v.shape[1] if hasattr(v, "shape") else len(v[0])
        print(f"[Ollama]     {ok(True)} - embedding '{modelo}' com {dim} dimensoes")
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
        print(f"[Groq]       {ok(True)} - chave presente (modelo {modelo}); use --testar-groq")
        return True
    try:
        import asyncio
        r = asyncio.run(_comum._llm_model_func("Responda apenas: ok", max_tokens=5))
        print(f"[Groq]       {ok(True)} - modelo {modelo} - resposta: {r[:40]!r}")
        return True
    except Exception as e:
        print(f"[Groq]       {ok(False)} - {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Checagem de ambiente da Aula 9 (LightRAG).")
    parser.add_argument("--testar-groq", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("  CHECAGEM DE AMBIENTE - Aula 9 (Graph RAG / LightRAG)")
    print("=" * 60)
    resultados = [
        checar_lightrag(),
        checar_corpus(),
        checar_ollama(),
        checar_groq(args.testar_groq),
    ]
    print("-" * 60)
    if all(resultados):
        print("Tudo pronto. Construa o grafo com: python 01_indexar_grafo.py")
    else:
        print("Resolva os itens FALHOU antes de seguir.")
    print(f"\nLLM da aula: {_comum.config_groq()[1]} | embeddings: {_comum.config_ollama()[1]}")
    print(f"Storage (arquivo): {_comum.WORKING_DIR}")


if __name__ == "__main__":
    main()
