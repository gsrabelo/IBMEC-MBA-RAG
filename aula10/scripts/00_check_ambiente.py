"""
00_check_ambiente.py - Confere se o ambiente da Aula 10 (Agentic/Adaptive RAG) esta pronto.

Verifica: Haystack Agent/Tool, Groq (LLM com tool-calling), Ollama (embeddings),
OpenSearch (indice do TCU), banco SQLite e (opcionais) Tavily e LangFuse.

Uso:
    python 00_check_ambiente.py
    python 00_check_ambiente.py --testar-groq
"""

import argparse

import _comum

_comum.carregar_env()


def ok(b):
    return "OK" if b else "FALHOU"


def checar_haystack_agent():
    try:
        from haystack.components.agents import Agent  # noqa: F401
        from haystack.tools import Tool  # noqa: F401
        print(f"[Haystack]   {ok(True)} - Agent + Tool disponiveis")
        return True
    except Exception as e:
        print(f"[Haystack]   {ok(False)} - {e}")
        return False


def checar_opensearch():
    try:
        n = _comum.abrir_store(_comum.INDICE_TCU).count_documents()
        print(f"[OpenSearch] {ok(True)} - indice '{_comum.INDICE_TCU}' com {n} documentos")
        if n == 0:
            print("             (vazio) rode a Aula 4 (01_indexar_hibrido) para popular o indice")
        return True
    except Exception as e:
        print(f"[OpenSearch] {ok(False)} - {e}")
        return False


def checar_ollama():
    try:
        emb = _comum.text_embedder()
        if hasattr(emb, "warm_up"):
            emb.warm_up()
        v = emb.run(text="teste")["embedding"]
        print(f"[Ollama]     {ok(True)} - embedding com {len(v)} dimensoes")
        return True
    except Exception as e:
        print(f"[Ollama]     {ok(False)} - {e}")
        return False


def checar_banco():
    if _comum.DB_PATH.exists():
        import sqlite3
        con = sqlite3.connect(_comum.DB_PATH)
        tabs = [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")]
        con.close()
        print(f"[SQLite]     {ok(True)} - {_comum.DB_PATH.name} (tabelas: {tabs})")
        return True
    print(f"[SQLite]     {ok(False)} - rode: python 01_preparar_dados.py")
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
        cliente, modelo = _comum.groq_client()
        r = _comum.gerar_texto(cliente, modelo, "Responda apenas: ok", max_tokens=5)
        print(f"[Groq]       {ok(True)} - resposta: {r!r}")
        return True
    except Exception as e:
        print(f"[Groq]       {ok(False)} - {e}")
        return False


def checar_opcionais():
    print(f"[Tavily]     {'OK' if _comum.tavily_configurado() else '(opcional) sem chave -> buscar_web offline'}")
    print(f"[LangFuse]   {'OK' if _comum.langfuse_configurado() else '(opcional) sem chaves -> sem tracing'}")


def main():
    parser = argparse.ArgumentParser(description="Checagem de ambiente da Aula 10.")
    parser.add_argument("--testar-groq", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("  CHECAGEM DE AMBIENTE - Aula 10 (Agentic / Adaptive RAG)")
    print("=" * 60)
    obrig = [checar_haystack_agent(), checar_opensearch(), checar_ollama(),
             checar_banco(), checar_groq(args.testar_groq)]
    checar_opcionais()
    print("-" * 60)
    print("Tudo pronto." if all(obrig) else "Resolva os itens FALHOU antes de seguir.")
    print(f"LLM: {_comum.config_groq()[1]} | embeddings: {_comum.config_ollama()[1]}")


if __name__ == "__main__":
    main()
