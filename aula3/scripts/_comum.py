"""
_comum.py - Funcoes auxiliares compartilhadas pelos scripts da Aula 3.

Evita repeticao de codigo: carregar .env, ler o corpus, configuracoes de
OpenSearch / Ollama / Groq / LangFuse, cliente Groq e reescrita de query.

Voce NAO executa este arquivo diretamente. Ele e importado pelos outros scripts.

IMPORTANTE: chame carregar_env() ANTES de importar o haystack nos scripts que
usam LangFuse, porque a auto-instrumentacao depende de uma variavel de ambiente
(HAYSTACK_CONTENT_TRACING_ENABLED) que precisa existir antes do import.
"""

import json
import os
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Caminhos importantes
# ---------------------------------------------------------------------------
PASTA_SCRIPTS = Path(__file__).resolve().parent           # .../aula3/scripts
PASTA_AULA3 = PASTA_SCRIPTS.parent                        # .../aula3
PASTA_PROJETO = PASTA_AULA3.parent                        # .../MBA_RAG_CAG
CORPUS = PASTA_AULA3 / "datasets" / "corpus_juridico_aula3.json"


def langfuse_configurado():
    """True se as chaves do LangFuse estao no ambiente."""
    return bool(os.getenv("LANGFUSE_SECRET_KEY") and os.getenv("LANGFUSE_PUBLIC_KEY"))


def carregar_env():
    """Carrega o .env e prepara o LangFuse (se configurado)."""
    caminho = None
    for c in [PASTA_PROJETO / ".env", PASTA_AULA3 / ".env", PASTA_SCRIPTS / ".env"]:
        if c.exists():
            load_dotenv(c)
            caminho = c
            break
    # A SDK do LangFuse usa LANGFUSE_HOST; no curso o .env usa LANGFUSE_BASE_URL.
    base = os.getenv("LANGFUSE_BASE_URL")
    if base and not os.getenv("LANGFUSE_HOST"):
        os.environ["LANGFUSE_HOST"] = base
    # Liga a auto-instrumentacao do Haystack quando o LangFuse esta configurado.
    if langfuse_configurado():
        os.environ["HAYSTACK_CONTENT_TRACING_ENABLED"] = "true"
    return caminho


def carregar_corpus():
    """Le o corpus da Aula 3. Devolve (documentos, queries_baseline).

    Cada documento ganha 'texto_completo' = ementa + texto (usado no embedding).
    """
    with open(CORPUS, "r", encoding="utf-8") as f:
        dados = json.load(f)
    documentos = dados.get("documentos", [])
    for d in documentos:
        d["texto_completo"] = f"{d.get('ementa', '')}. {d.get('texto', '')}".strip()
    return documentos, dados.get("queries_baseline", [])


# ---------------------------------------------------------------------------
# Configuracoes (.env com valores padrao)
# ---------------------------------------------------------------------------
def config_ollama():
    return (os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text"))


def config_opensearch():
    host = os.getenv("OPENSEARCH_HOST", "localhost")
    porta = os.getenv("OPENSEARCH_PORT", "9200")
    usuario = os.getenv("OPENSEARCH_USER", "")
    senha = os.getenv("OPENSEARCH_PASS", "")
    return {"url": f"http://{host}:{porta}", "usuario": usuario, "senha": senha}


def config_groq():
    return (os.getenv("GROQ_API_KEY", ""),
            os.getenv("GROQ_LLM_MODEL", "llama-3.1-8b-instant"),
            "https://api.groq.com/openai/v1")


DIMENSAO_EMBEDDING = {"nomic-embed-text": 768, "mxbai-embed-large": 1024, "bge-m3": 1024}


def dimensao_do_modelo(nome_modelo):
    return DIMENSAO_EMBEDDING.get(nome_modelo.split(":")[0].lower(), 768)


# ---------------------------------------------------------------------------
# LLM (Groq) e Query Rewriting - usados pelos scripts 01 e 04
# ---------------------------------------------------------------------------
def groq_client():
    """Devolve (cliente OpenAI apontando para a Groq, modelo)."""
    from openai import OpenAI

    api_key, modelo, base_url = config_groq()
    return OpenAI(api_key=api_key, base_url=base_url), modelo


# Prompts das 3 tecnicas de reescrita de query (Aula 3).
PROMPTS_REWRITE = {
    "paraphrase": (
        "Voce e um especialista juridico brasileiro em processo penal. "
        "Reformule a pergunta abaixo usando terminologia tecnica juridica, "
        "mantendo o mesmo significado. Retorne APENAS a pergunta reformulada.\n\nPergunta: {q}"
    ),
    "hyde": (
        "Voce e um redator juridico. Escreva um paragrafo curto (4-6 linhas) que seria "
        "encontrado em um acordao, codigo ou doutrina brasileira e que responderia a pergunta. "
        "Cite artigos de lei quando apropriado. Retorne APENAS o paragrafo.\n\nPergunta: {q}"
    ),
    "stepback": (
        "Voce e um especialista em direito processual penal. Dada a pergunta especifica, "
        "formule uma pergunta MAIS GERAL sobre o conceito juridico por tras dela. "
        "Retorne APENAS a pergunta geral.\n\nPergunta especifica: {q}"
    ),
}


def reescrever_query(cliente, modelo, query, tecnica):
    """Reescreve a query com uma das tecnicas: paraphrase, hyde ou stepback."""
    prompt = PROMPTS_REWRITE[tecnica].format(q=query)
    resp = cliente.chat.completions.create(
        model=modelo,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=300,
    )
    return resp.choices[0].message.content.strip()
