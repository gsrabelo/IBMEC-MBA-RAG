"""
00_check_ambiente.py - Verifica se o ambiente da Aula 7 esta pronto.

Testa: Python, bibliotecas (haystack + joiner RRF, langfuse, ollama), .env,
corpus do TCU (Aula 4), OpenSearch + indice reaproveitado, Ollama, Groq, LangFuse.

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
        "haystack_integrations.document_stores.opensearch",
        "haystack.components.joiners",  # DocumentJoiner (RRF)
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


def checar_corpus():
    print("4) Corpus do TCU (Aula 4)")
    try:
        docs = _comum.carregar_acordaos_aula4(limite=3)
        ok(f"corpus do TCU acessivel ({_comum.CORPUS_ACORDAOS_AULA4.name})")
    except Exception as e:
        falha(f"nao consegui ler o corpus do TCU -> {e}")
        aviso("rode a Aula 4 antes (ela tem o corpus_juridico_aula4_v2.json)")


def checar_opensearch():
    print("5) OpenSearch + indice reaproveitado")
    cfg = _comum.config_opensearch()
    try:
        import requests

        auth = (cfg["usuario"], cfg["senha"]) if cfg["usuario"] else None
        r = requests.get(cfg["url"], auth=auth, timeout=5)
        r.raise_for_status()
        ok(f"OpenSearch em {cfg['url']} (versao {r.json().get('version', {}).get('number', '?')})")
        try:
            store = _comum.abrir_store(_comum.INDICE_TCU)
            n = store.count_documents()
            (ok if n > 0 else aviso)(f"indice '{_comum.INDICE_TCU}': {n} documentos "
                                     + ("(reaproveitado)" if n > 0 else "(vazio - rode 01_indexar_opensearch.py)"))
        except Exception as e:
            aviso(f"nao consegui contar o indice '{_comum.INDICE_TCU}': {e}")
    except Exception as e:
        falha(f"OpenSearch nao respondeu em {cfg['url']} -> {e}")


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
    print("7) Groq (LLM)")
    api_key, modelo, base_url = _comum.config_groq()
    if not api_key:
        falha("GROQ_API_KEY ausente no .env")
        return
    ok(f"GROQ_API_KEY presente | modelo padrao: {modelo}")
    if not testar:
        aviso("use --testar-groq para uma chamada real de teste")
        return
    try:
        cliente, _ = _comum.groq_client()
        r = _comum.gerar_texto(cliente, modelo, "Responda apenas: OK", max_tokens=5)
        ok(f"chamada Groq funcionou -> {r}")
    except Exception as e:
        falha(f"chamada Groq falhou -> {e}")


def checar_langfuse():
    print("8) LangFuse (observabilidade)")
    import os

    if not _comum.langfuse_configurado():
        aviso("chaves do LangFuse ausentes - o chat (06) roda sem tracing")
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
    parser = argparse.ArgumentParser(description="Verifica o ambiente da Aula 7.")
    parser.add_argument("--testar-groq", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("  CHECAGEM DE AMBIENTE - MBA RAG & CAG - Aula 7")
    print("=" * 60)
    checar_python()
    checar_bibliotecas()
    checar_env()
    checar_corpus()
    checar_opensearch()
    checar_ollama()
    checar_groq(args.testar_groq)
    checar_langfuse()
    print("=" * 60)
    print("  Fim da checagem. Resolva os itens [FALHA] antes de seguir.")
    print("=" * 60)


if __name__ == "__main__":
    main()
