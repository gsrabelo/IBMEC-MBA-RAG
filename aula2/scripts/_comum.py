"""
_comum.py - Funcoes auxiliares compartilhadas pelos scripts da Aula 2.

Evita repeticao de codigo (carregar .env, achar a pasta de datasets, ler o corpus
de exemplo, pegar configuracoes de OpenSearch / Ollama / Groq). E proposital ser simples.

Voce NAO executa este arquivo diretamente. Ele e importado pelos outros scripts.
"""

import json
import os
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Caminhos importantes (calculados a partir da localizacao deste arquivo)
# ---------------------------------------------------------------------------
PASTA_SCRIPTS = Path(__file__).resolve().parent           # .../aula2/scripts
PASTA_AULA2 = PASTA_SCRIPTS.parent                        # .../aula2
PASTA_PROJETO = PASTA_AULA2.parent                        # .../MBA_RAG_CAG
PASTA_DATASETS = PASTA_AULA2 / "datasets"                 # .../aula2/datasets
CORPUS_SAMPLE = PASTA_DATASETS / "corpus_juridico_sample.json"

# PDFs de exemplo da Aula 2 (existem na pasta datasets).
PDF_DIGITAL = PASTA_DATASETS / "Manual_DPCA_atualizado.pdf"   # PDF com texto nativo
PDF_ESCANEADO = PASTA_DATASETS / "Laudo-Minimal.pdf"         # PDF imagem (precisa OCR)


def carregar_env():
    """Procura um arquivo .env no projeto e carrega as variaveis de ambiente."""
    for caminho in [PASTA_PROJETO / ".env", PASTA_AULA2 / ".env", PASTA_SCRIPTS / ".env"]:
        if caminho.exists():
            load_dotenv(caminho)
            return caminho
    return None


def carregar_corpus_sample():
    """Le o corpus de exemplo (datasets/corpus_juridico_sample.json).

    Devolve (documentos, perguntas_teste). Cada documento ganha 'texto'.
    """
    with open(CORPUS_SAMPLE, "r", encoding="utf-8") as f:
        dados = json.load(f)
    documentos = dados.get("documentos", [])
    return documentos, dados.get("perguntas_teste", [])


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
    url = f"http://{host}:{porta}"   # lab: single-node sem HTTPS
    return {"url": url, "usuario": usuario, "senha": senha}


def config_groq():
    """Devolve (api_key, modelo_llm, base_url) da Groq (API compativel com OpenAI)."""
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
