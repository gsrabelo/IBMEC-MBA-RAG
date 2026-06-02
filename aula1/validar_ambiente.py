#!/usr/bin/env python3
"""
Validação do Ambiente — MBA RAG & CAG Aula 1
Execute com: python validar_ambiente.py
"""
import sys, requests
from datetime import datetime

resultados = []

def teste(nome, fn, opcional=False):
    try:
        fn()
        resultados.append(('OK', nome, ''))
    except Exception as e:
        if opcional:
            resultados.append(('WARN', f'{nome} (opcional)', str(e)[:80]))
        else:
            resultados.append(('FAIL', nome, str(e)[:80]))

print('=' * 65)
print(f'VALIDACAO DO AMBIENTE MBA RAG — {datetime.now().strftime("%d/%m/%Y %H:%M")}')
print('=' * 65)

# Python 3.11+
def check_python():
    assert sys.version_info >= (3, 11), f"Python 3.11+ necessario, encontrado: {sys.version}"
teste("Python 3.11+", check_python)

# Ollama rodando
def check_ollama():
    r = requests.get("http://localhost:11434/api/tags", timeout=5)
    assert r.status_code == 200
    modelos = [m["name"] for m in r.json().get("models", [])]
    assert len(modelos) > 0, "Nenhum modelo instalado. Execute: ollama pull llama3.2:3b"
    print(f"\n     Modelos: {modelos}", end="")
teste("Ollama (servidor + modelos)", check_ollama)

# Modelo LLM via Ollama
def check_llm():
    r = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": "llama3.1:8b", "prompt": "OK", "stream": False},
        timeout=60
    )
    assert r.status_code == 200
    assert len(r.json()["response"]) > 0
teste("Ollama LLM (llama3.1:8b geracao)", check_llm)

# Embedding via Ollama
def check_embed():
    r = requests.post(
        "http://localhost:11434/api/embeddings",
        json={"model": "mxbai-embed-large", "prompt": "crime de furto"},
        timeout=30
    )
    assert r.status_code == 200
    dims = len(r.json()["embedding"])
    assert dims > 0
    print(f"\n     Dimensoes: {dims}", end="")
teste("Ollama Embeddings (mxbai-embed-large)", check_embed)

# OpenSearch
def check_opensearch():
    r = requests.get("http://localhost:9200", timeout=5)
    assert r.status_code == 200
    assert "version" in r.json()
teste("OpenSearch 3.x (servidor local)", check_opensearch)

# Bibliotecas Python
def check_libs():
    import sentence_transformers, faiss, opensearchpy, langchain, langfuse, umap
teste("Bibliotecas Python (sentence-transformers, faiss, langchain)", check_libs)

# LangFuse
def check_langfuse():
    import os
    from dotenv import load_dotenv
    from langfuse import Langfuse
    load_dotenv()
    pk = os.environ.get("LANGFUSE_PUBLIC_KEY", "")
    sk = os.environ.get("LANGFUSE_SECRET_KEY", "")
    print(f"Public key: {pk}")
    print(f"Secret key: {sk}")
    assert pk and not pk.endswith("_AQUI"), "Configure LANGFUSE_PUBLIC_KEY no .env"
    print(f"Public key: {pk}")
    lf = Langfuse(public_key=pk, secret_key=sk)
    lf.auth_check()
teste("LangFuse (observabilidade)", check_langfuse)

# Kernel Jupyter
def check_jupyter():
    import ipykernel
    result = __import__("subprocess").run(
        [sys.executable, "-m", "jupyter", "kernelspec", "list"],
        capture_output=True, text=True
    )
    assert ".venv" in result.stdout or "mba-rag" in result.stdout.lower(), \
        "Kernel '.venv' nao encontrado. Execute: python -m ipykernel install --user --name=.venv"
teste("Kernel Jupyter registrado (.venv)", check_jupyter)

# Imprimir resultados
print()
for status, nome, detalhe in resultados:
    icone = {"OK": "[OK]", "WARN": "[AV]", "FAIL": "[XX]"}[status]
    print(f"  {icone}  {nome}")
    if detalhe and status == "FAIL":
        print(f"       -> {detalhe}")

ok = sum(1 for s, _, _ in resultados if s == "OK")
total = len(resultados)
print()
print(f"Resultado: {ok}/{total} verificacoes aprovadas")

falhas = [n for s, n, _ in resultados if s == "FAIL"]
if not falhas:
    print()
    print("AMBIENTE COMPLETAMENTE VALIDADO!")
    print("Voce esta pronto para os laboratorios da Aula 1.")
else:
    print()
    print("Falhas detectadas — resolva antes de continuar:")
    for f in falhas:
        print(f"  - {f}")
