"""
_comum.py - Funcoes auxiliares compartilhadas pelos scripts da Aula 1.

Objetivo: evitar repeticao de codigo nos scripts (carregar .env, ler o corpus,
pegar configuracoes de OpenSearch / Ollama / Groq). E proposital ser simples.

Voce NAO executa este arquivo diretamente. Ele e importado pelos outros scripts.
"""

import json
import os
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Caminhos importantes (calculados a partir da localizacao deste arquivo)
# ---------------------------------------------------------------------------
PASTA_SCRIPTS = Path(__file__).resolve().parent          # .../aula1/scripts
PASTA_AULA1 = PASTA_SCRIPTS.parent                       # .../aula1
PASTA_PROJETO = PASTA_AULA1.parent                       # .../MBA_RAG_CAG
CAMINHO_CORPUS = PASTA_AULA1 / "datasets" / "corpus_juridico_aula1.json"


def carregar_env():
    """Procura um arquivo .env no projeto e carrega as variaveis de ambiente.

    Ordem de busca: raiz do projeto -> pasta da aula1 -> pasta de scripts.
    """
    for caminho in [PASTA_PROJETO / ".env", PASTA_AULA1 / ".env", PASTA_SCRIPTS / ".env"]:
        if caminho.exists():
            load_dotenv(caminho)
            return caminho
    return None


def carregar_corpus():
    """Le o corpus juridico (lista de documentos) e devolve uma lista de dicts.

    Cada documento ganha o campo 'texto' = titulo + conteudo, que e o que
    realmente vai para o embedding.
    """
    with open(CAMINHO_CORPUS, "r", encoding="utf-8") as f:
        corpus = json.load(f)
    for doc in corpus:
        doc["texto"] = f"{doc.get('titulo', '')}. {doc.get('conteudo', '')}".strip()
    return corpus


# ---------------------------------------------------------------------------
# Configuracoes lidas do .env (com valores padrao sensatos)
# ---------------------------------------------------------------------------
def config_ollama():
    """Devolve (base_url, modelo_embedding) do Ollama."""
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    modelo = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
    return base_url, modelo


def config_opensearch():
    """Devolve um dicionario com a configuracao de conexao do OpenSearch."""
    host = os.getenv("OPENSEARCH_HOST", "localhost")
    porta = os.getenv("OPENSEARCH_PORT", "9200")
    usuario = os.getenv("OPENSEARCH_USER", "")
    senha = os.getenv("OPENSEARCH_PASS", "")
    # No lab usamos OpenSearch single-node SEM HTTPS (security desabilitada).
    url = f"http://{host}:{porta}"
    return {"url": url, "usuario": usuario, "senha": senha}


def config_groq():
    """Devolve (api_key, modelo_llm, base_url) da Groq.

    A Groq expoe uma API compativel com a da OpenAI, por isso usamos a base_url
    /openai/v1 junto com o OpenAIGenerator do Haystack.
    """
    api_key = os.getenv("GROQ_API_KEY", "")
    modelo = os.getenv("GROQ_LLM_MODEL", "llama-3.1-8b-instant")
    base_url = "https://api.groq.com/openai/v1"
    return api_key, modelo, base_url


# Dimensao do vetor por modelo de embedding (usada ao criar o indice kNN).
DIMENSAO_EMBEDDING = {
    "nomic-embed-text": 768,
    "mxbai-embed-large": 1024,
    "bge-m3": 1024,
}


def dimensao_do_modelo(nome_modelo):
    """Descobre a dimensao do vetor a partir do nome do modelo de embedding."""
    chave = nome_modelo.split(":")[0].lower()
    return DIMENSAO_EMBEDDING.get(chave, 768)
