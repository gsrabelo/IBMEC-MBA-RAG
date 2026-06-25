"""
00_check_ambiente.py - Verifica se o ambiente da Aula 5 esta pronto.

Testa (um item por linha, com OK ou FALHA):
  1. Python
  2. Bibliotecas (ragas, deepeval, langchain_groq/ollama, langfuse, haystack, etc.)
  3. .env e chaves
  4. Perguntas de avaliacao
  5. OpenSearch + kNN (indice que sera avaliado)
  6. Ollama + modelo de embedding
  7. Groq (chave; opcional teste real)
  8. LangFuse (chaves + servidor, se configurado)

Uso:
    python 00_check_ambiente.py
    python 00_check_ambiente.py --testar-groq
"""

import argparse
import sys

import _comum


def ok(m): print(f"  [ OK ]   {m}")
def falha(m): print(f"  [FALHA]  {m}")
def aviso(m): print(f"  [ ! ]    {m}")


def checar_python():
    print("1) Python")
    v = sys.version_info
    (ok if v >= (3, 10) else falha)(f"Python {v.major}.{v.minor}.{v.micro}")


def checar_bibliotecas():
    print("2) Bibliotecas")
    libs = [
        "haystack",
        "ragas",
        "deepeval",
        "langchain_groq",
        "langchain_ollama",
        "langfuse",
        "ollama",
        "dotenv",
    ]
    for lib in libs:
        try:
            __import__(lib)
            ok(lib)
        except Exception as e:
            falha(f"{lib} -> {e}")


def checar_env():
    print("3) Arquivo .env")
    caminho = _comum.carregar_env()
    if caminho is None:
        falha("Nenhum .env encontrado")
        return
    ok(f".env carregado de: {caminho}")
    import os

    for chave in ["GROQ_API_KEY", "OLLAMA_BASE_URL", "OPENSEARCH_HOST"]:
        (ok if os.getenv(chave) else aviso)(
            f"variavel {chave} {'definida' if os.getenv(chave) else 'ausente (usarei padrao)'}")


def checar_perguntas():
    print("4) Perguntas de avaliacao")
    try:
        p = _comum.carregar_perguntas()
        ok(f"{len(p)} perguntas com ground_truth")
    except Exception as e:
        falha(f"nao consegui ler as perguntas -> {e}")


def checar_opensearch():
    print("5) OpenSearch")
    cfg = _comum.config_opensearch()
    try:
        import requests

        auth = (cfg["usuario"], cfg["senha"]) if cfg["usuario"] else None
        r = requests.get(cfg["url"], auth=auth, timeout=5)
        r.raise_for_status()
        ok(f"OpenSearch em {cfg['url']} (versao {r.json().get('version', {}).get('number', '?')})")
    except Exception as e:
        falha(f"OpenSearch nao respondeu em {cfg['url']} -> {e}")
        aviso("o 01 precisa de um indice ja criado (ex.: aula4_hibrido). Veja a Aula 4.")


def checar_ollama():
    print("6) Ollama")
    base_url, modelo = _comum.config_ollama()
    try:
        import requests

        r = requests.get(f"{base_url}/api/tags", timeout=5)
        r.raise_for_status()
        modelos = [m["name"] for m in r.json().get("models", [])]
        ok(f"Ollama em {base_url}")
        (ok if any(modelo in m for m in modelos) else falha)(
            f"modelo de embedding '{modelo}' " + ("baixado" if any(modelo in m for m in modelos) else "ausente (ollama pull)"))
    except Exception as e:
        falha(f"Ollama nao respondeu em {base_url} -> {e}")


def checar_groq(testar):
    print("7) Groq (LLM judge)")
    api_key, modelo, base_url = _comum.config_groq()
    if not api_key:
        falha("GROQ_API_KEY ausente no .env")
        return
    ok(f"GROQ_API_KEY presente | modelo padrao: {modelo}")
    if not testar:
        aviso("use --testar-groq para uma chamada real de teste")
        return
    try:
        from openai import OpenAI

        cliente = OpenAI(api_key=api_key, base_url=base_url)
        resp = cliente.chat.completions.create(
            model=modelo, messages=[{"role": "user", "content": "Responda apenas: OK"}], max_tokens=5)
        ok(f"chamada Groq funcionou -> {resp.choices[0].message.content.strip()}")
    except Exception as e:
        falha(f"chamada Groq falhou -> {e}")


def checar_langfuse():
    print("8) LangFuse (Scores API)")
    import os

    if not _comum.langfuse_configurado():
        aviso("chaves do LangFuse ausentes - o 04 (envio de scores) precisara delas")
        aviso("veja GUIA_LANGFUSE_WINDOWS.md da Aula 3")
        return
    host = os.getenv("LANGFUSE_HOST", "http://localhost:3000")
    ok(f"chaves presentes | host: {host}")
    try:
        import requests

        r = requests.get(f"{host}/api/public/health", timeout=5)
        (ok if r.status_code == 200 else aviso)(f"servidor LangFuse respondeu ({r.status_code})")
    except Exception as e:
        aviso(f"servidor LangFuse nao respondeu em {host} -> {e}")


def main():
    parser = argparse.ArgumentParser(description="Verifica o ambiente da Aula 5.")
    parser.add_argument("--testar-groq", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("  CHECAGEM DE AMBIENTE - MBA RAG & CAG - Aula 5")
    print("=" * 60)
    checar_python()
    checar_bibliotecas()
    checar_env()
    checar_perguntas()
    checar_opensearch()
    checar_ollama()
    checar_groq(args.testar_groq)
    checar_langfuse()
    print("=" * 60)
    print("  Fim da checagem. Resolva os itens [FALHA] antes de seguir.")
    print("=" * 60)


if __name__ == "__main__":
    main()
