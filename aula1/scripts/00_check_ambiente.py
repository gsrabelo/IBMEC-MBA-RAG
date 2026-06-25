"""
00_check_ambiente.py - Verifica se o ambiente da Aula 1 esta pronto.

O que ele testa (um item por linha, com OK ou FALHA):
  1. Versao do Python
  2. Bibliotecas principais instaladas (haystack, ollama, etc.)
  3. Arquivo .env encontrado e chaves esperadas
  4. OpenSearch respondendo + plugin kNN disponivel
  5. Ollama respondendo + modelo de embedding baixado
  6. Chave da Groq presente (e, opcionalmente, um teste real de chamada)

Uso:
    python 00_check_ambiente.py            # checagem padrao
    python 00_check_ambiente.py --testar-groq   # tambem faz 1 chamada real na Groq
"""

import argparse
import sys

import _comum


def ok(msg):
    print(f"  [ OK ]   {msg}")


def falha(msg):
    print(f"  [FALHA]  {msg}")


def aviso(msg):
    print(f"  [ ! ]    {msg}")


def checar_python():
    print("1) Python")
    versao = sys.version_info
    if versao >= (3, 10):
        ok(f"Python {versao.major}.{versao.minor}.{versao.micro}")
    else:
        falha(f"Python {versao.major}.{versao.minor} - o curso pede 3.10 ou superior")


def checar_bibliotecas():
    print("2) Bibliotecas")
    libs = [
        "haystack",
        "haystack_integrations.document_stores.opensearch",
        "haystack_integrations.components.embedders.ollama",
        "opensearchpy",
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
        falha("Nenhum .env encontrado (esperado na raiz do projeto)")
        return
    ok(f".env carregado de: {caminho}")
    import os

    for chave in ["GROQ_API_KEY", "OLLAMA_BASE_URL", "OPENSEARCH_HOST"]:
        if os.getenv(chave):
            ok(f"variavel {chave} definida")
        else:
            aviso(f"variavel {chave} ausente (usarei o valor padrao)")


def checar_opensearch():
    print("4) OpenSearch")
    cfg = _comum.config_opensearch()
    try:
        import requests

        auth = (cfg["usuario"], cfg["senha"]) if cfg["usuario"] else None
        r = requests.get(cfg["url"], auth=auth, timeout=5)
        r.raise_for_status()
        versao = r.json().get("version", {}).get("number", "?")
        ok(f"OpenSearch respondendo em {cfg['url']} (versao {versao})")

        # Verifica se o plugin kNN esta instalado
        r2 = requests.get(f"{cfg['url']}/_cat/plugins?format=json", auth=auth, timeout=5)
        plugins = [p.get("component", "") for p in r2.json()]
        if any("knn" in p.lower() for p in plugins):
            ok("plugin kNN disponivel")
        else:
            aviso("plugin kNN NAO encontrado - busca vetorial pode falhar")
    except Exception as e:
        falha(f"OpenSearch nao respondeu em {cfg['url']} -> {e}")
        aviso("Veja o GUIA_OPENSEARCH_WINDOWS.md para subir o servidor")


def checar_ollama():
    print("5) Ollama")
    base_url, modelo = _comum.config_ollama()
    try:
        import requests

        r = requests.get(f"{base_url}/api/tags", timeout=5)
        r.raise_for_status()
        modelos = [m["name"] for m in r.json().get("models", [])]
        ok(f"Ollama respondendo em {base_url}")
        if any(modelo in m for m in modelos):
            ok(f"modelo de embedding '{modelo}' baixado")
        else:
            falha(f"modelo '{modelo}' nao encontrado. Rode: ollama pull {modelo}")
            aviso(f"modelos disponiveis: {modelos}")
    except Exception as e:
        falha(f"Ollama nao respondeu em {base_url} -> {e}")
        aviso("Inicie o Ollama (ollama serve) e baixe o modelo de embedding")


def checar_groq(testar):
    print("6) Groq (LLM)")
    api_key, modelo, base_url = _comum.config_groq()
    if not api_key:
        falha("GROQ_API_KEY ausente no .env")
        return
    ok(f"GROQ_API_KEY presente | modelo padrao: {modelo}")
    if not testar:
        aviso("use --testar-groq para fazer uma chamada real de teste")
        return
    try:
        from openai import OpenAI

        cliente = OpenAI(api_key=api_key, base_url=base_url)
        resp = cliente.chat.completions.create(
            model=modelo,
            messages=[{"role": "user", "content": "Responda apenas: OK"}],
            max_tokens=5,
        )
        ok(f"chamada Groq funcionou -> resposta: {resp.choices[0].message.content.strip()}")
    except Exception as e:
        falha(f"chamada Groq falhou -> {e}")


def main():
    parser = argparse.ArgumentParser(description="Verifica o ambiente da Aula 1.")
    parser.add_argument("--testar-groq", action="store_true",
                        help="faz uma chamada real na Groq para validar a chave")
    args = parser.parse_args()

    print("=" * 60)
    print("  CHECAGEM DE AMBIENTE - MBA RAG & CAG - Aula 1")
    print("=" * 60)
    checar_python()
    checar_bibliotecas()
    checar_env()
    checar_opensearch()
    checar_ollama()
    checar_groq(args.testar_groq)
    print("=" * 60)
    print("  Fim da checagem. Resolva os itens [FALHA] antes de seguir.")
    print("=" * 60)


if __name__ == "__main__":
    main()
