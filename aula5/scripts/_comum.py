"""
_comum.py - Funcoes auxiliares compartilhadas pelos scripts da Aula 5.

Aula 5 = Avaliacao e Observabilidade (RAGAS, DeepEval, LangFuse Scores).

Centraliza: carregar .env, ler as perguntas de avaliacao, ler os acordaos do
corpus (para GERAR perguntas), configuracoes de OpenSearch / Ollama / Groq /
LangFuse e os "judges" LangChain (Groq + Ollama) usados pelo RAGAS e DeepEval.

Voce NAO executa este arquivo diretamente. Ele e importado pelos outros scripts.
"""

import json
import os
from pathlib import Path

from dotenv import load_dotenv

PASTA_SCRIPTS = Path(__file__).resolve().parent           # .../aula5/scripts
PASTA_AULA5 = PASTA_SCRIPTS.parent                        # .../aula5
PASTA_PROJETO = PASTA_AULA5.parent                        # .../MBA_RAG_CAG

# Perguntas "genericas" que vem com a aula (direito penal/processual - nao casam
# com o corpus do TCU). Mantidas como alternativa.
PERGUNTAS_ORIGINAIS = PASTA_AULA5 / "datasets" / "corpus_avaliacao_aula5.json"
# Perguntas GERADAS a partir dos acordaos (script 01) - casam com o indice TCU.
PERGUNTAS_GERADAS = PASTA_SCRIPTS / "perguntas_geradas.json"
# Corpus de acordaos do TCU (Aula 4) usado para gerar perguntas.
CORPUS_ACORDAOS = PASTA_PROJETO / "aula4" / "datasets" / "corpus_juridico_aula4_v2.json"

# arquivos intermediarios
DATASET_COMPLETO = PASTA_SCRIPTS / "dataset_avaliacao_completo.json"
RESULTADOS_RAGAS = PASTA_SCRIPTS / "ragas_resultados.json"


def langfuse_configurado():
    return bool(os.getenv("LANGFUSE_SECRET_KEY") and os.getenv("LANGFUSE_PUBLIC_KEY"))


def carregar_env():
    caminho = None
    for c in [PASTA_PROJETO / ".env", PASTA_AULA5 / ".env", PASTA_SCRIPTS / ".env"]:
        if c.exists():
            load_dotenv(c)
            caminho = c
            break
    base = os.getenv("LANGFUSE_BASE_URL")
    if base and not os.getenv("LANGFUSE_HOST"):
        os.environ["LANGFUSE_HOST"] = base
    if langfuse_configurado():
        os.environ["HAYSTACK_CONTENT_TRACING_ENABLED"] = "true"
    return caminho


def carregar_perguntas(caminho=None, limite=0):
    """Le as perguntas de avaliacao [{id, question, ground_truth, tipo}].

    Por padrao usa as perguntas GERADAS (perguntas_geradas.json) se existirem;
    senao, cai para as perguntas originais da aula. Use 'caminho' para forcar.
    """
    if caminho is None:
        caminho = PERGUNTAS_GERADAS if PERGUNTAS_GERADAS.exists() else PERGUNTAS_ORIGINAIS
    with open(caminho, "r", encoding="utf-8") as f:
        perguntas = json.load(f)
    return perguntas[:limite] if (limite and limite > 0) else perguntas


def carregar_acordaos(limite=0):
    """Le os acordaos do TCU (Aula 4). Cada doc ganha 'texto' = titulo + conteudo."""
    with open(CORPUS_ACORDAOS, "r", encoding="utf-8") as f:
        docs = json.load(f)
    if limite and limite > 0:
        docs = docs[:limite]
    for d in docs:
        d["texto"] = f"{d.get('titulo', '')}. {d.get('conteudo', '')}".strip()
    return docs


def carregar_dataset_completo():
    """Le o dataset gerado pelo 02 (question, contexts, answer, ground_truth)."""
    if not DATASET_COMPLETO.exists():
        raise FileNotFoundError(
            f"{DATASET_COMPLETO.name} nao encontrado. Rode primeiro: python 02_gerar_dataset.py")
    with open(DATASET_COMPLETO, "r", encoding="utf-8") as f:
        return json.load(f)


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


def groq_client():
    """Devolve (cliente OpenAI apontando para a Groq, modelo)."""
    from openai import OpenAI

    api_key, modelo, base_url = config_groq()
    return OpenAI(api_key=api_key, base_url=base_url), modelo


# ---------------------------------------------------------------------------
# "Judges" LangChain - usados como LLM avaliador (RAGAS e DeepEval)
# ---------------------------------------------------------------------------
def chat_groq(temperature=0.0):
    """Devolve (ChatGroq, nome_modelo). A chave vem do .env (GROQ_API_KEY)."""
    from langchain_groq import ChatGroq

    _, modelo, _ = config_groq()
    return ChatGroq(model=modelo, temperature=temperature), modelo


def ollama_embeddings_lc():
    """Devolve um OllamaEmbeddings (LangChain) para o RAGAS."""
    from langchain_ollama import OllamaEmbeddings

    base_url, modelo = config_ollama()
    return OllamaEmbeddings(model=modelo, base_url=base_url)
