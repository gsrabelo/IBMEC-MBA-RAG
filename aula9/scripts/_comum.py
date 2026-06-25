"""
_comum.py - Funcoes auxiliares compartilhadas pelos scripts da Aula 9.

Aula 9 = Graph RAG com LightRAG.
Stack: LightRAG (grafo de conhecimento) + Groq (LLM) + Ollama (embeddings) +
storage NATIVO em arquivo (NetworkX + NanoVectorDB + JSON) no working_dir.

NAO usa OpenSearch nem LangFuse nesta aula (LightRAG nao e Haystack). O LightRAG
guarda TUDO em arquivos dentro de 'aula9/rag_storage/':
  - graph_chunk_entity_relation.graphml  (o GRAFO, via NetworkX)
  - vdb_entities.json / vdb_relationships.json / vdb_chunks.json (vetores, NanoVectorDB)
  - kv_store_*.json (chunks, docs, cache de LLM)

IMPORTANTE: o LightRAG e ASSINCRONO. Os scripts usam asyncio.run(...).

Voce NAO executa este arquivo diretamente. Ele e importado pelos outros scripts.
"""

import os
from functools import partial
from pathlib import Path

from dotenv import load_dotenv

PASTA_SCRIPTS = Path(__file__).resolve().parent
PASTA_AULA9 = PASTA_SCRIPTS.parent
PASTA_PROJETO = PASTA_AULA9.parent

CORPUS = PASTA_AULA9 / "datasets" / "corpus_juridico.txt"
WORKING_DIR = PASTA_AULA9 / "rag_storage"   # onde o LightRAG persiste grafo+vetores+kv
GRAPHML = WORKING_DIR / "graph_chunk_entity_relation.graphml"

# dimensao do modelo de embedding do Ollama (nomic-embed-text = 768)
DIMENSAO_EMBEDDING = {"nomic-embed-text": 768, "mxbai-embed-large": 1024, "bge-m3": 1024}


def carregar_env():
    """Carrega o .env do projeto (procura na raiz, na pasta da aula e em scripts)."""
    for c in [PASTA_PROJETO / ".env", PASTA_AULA9 / ".env", PASTA_SCRIPTS / ".env"]:
        if c.exists():
            load_dotenv(c)
            return c
    return None


def config_groq():
    """(api_key, modelo, base_url) para a Groq (OpenAI-compatible).

    LLM da Aula 9: por padrao llama-3.3-70b-versatile. A doc do LightRAG DESACONSELHA
    modelos de raciocinio (gpt-oss etc.) na fase de extracao de entidades - por isso
    NAO usamos GROQ_LLM_MODEL aqui; use AULA9_LLM_MODEL se quiser trocar.
    """
    return (os.getenv("GROQ_API_KEY", ""),
            os.getenv("GROQ_LLM_MODEL", "llama-3.3-70b-versatile"),
            os.getenv("GROQ_LLM_HOST", "https://api.groq.com/openai/v1"))


def config_ollama():
    """(base_url, modelo) do Ollama para embeddings.

    Aula 9 padroniza em bge-m3 (1024 dim) - o MESMO modelo do LightRAG Server/WebUI,
    para que o grafo construido pelos scripts seja aberto/visualizado no servidor.
    Troque com AULA9_EMBED_MODEL no .env (precisa re-indexar com --recriar se mudar).
    """
    return (os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            os.getenv("AULA9_EMBED_MODEL", "bge-m3:latest"))


def dimensao_do_modelo(nome_modelo):
    return DIMENSAO_EMBEDDING.get(nome_modelo.split(":")[0].lower(), 768)


def ler_corpus():
    if not CORPUS.exists():
        raise FileNotFoundError(f"Corpus nao encontrado: {CORPUS}")
    return CORPUS.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# LightRAG: LLM (Groq) e embeddings (Ollama)
# ---------------------------------------------------------------------------
async def _llm_model_func(prompt, system_prompt=None, history_messages=None, **kwargs) -> str:
    """ModelFunc para o LightRAG chamando a API do Groq (usando a lib oficial do Groq via API OpenAI-compatible)."""
    from lightrag.llm.openai import openai_complete_if_cache

    # --- INJETADO: Força o comportamento em português direto no System Prompt ---
    instrucao_pt = "\n\nCRITICAL: Return all entity descriptions, relationship summaries, and metadata strictly in Portuguese (Brazil). Even if the prompt parameters are in English, your textual output inside JSON or text blocks must be in Portuguese."
    if system_prompt:
        system_prompt = system_prompt + instrucao_pt
    else:
        system_prompt = "You are a helpful assistant. " + instrucao_pt

    api_key, modelo, base_url = config_groq()
    return await openai_complete_if_cache(
        modelo, prompt, system_prompt=system_prompt,
        history_messages=history_messages or [],
        api_key=api_key, base_url=base_url, **kwargs)


def _embedding_func():
    """EmbeddingFunc do LightRAG usando o Ollama (nomic-embed-text)."""
    from lightrag.llm.ollama import ollama_embed
    from lightrag.utils import EmbeddingFunc

    base_url, modelo = config_ollama()
    return EmbeddingFunc(
        embedding_dim=dimensao_do_modelo(modelo),
        max_token_size=8192,
        func=partial(ollama_embed.func, embed_model=modelo, host=base_url),
    )


async def criar_rag(working_dir=None):
    """Cria e inicializa uma instancia LightRAG com storage NATIVO em arquivo.

    Storage padrao do LightRAG (NetworkX + NanoVectorDB + JSON) - nada para instalar.
    """
    from lightrag import LightRAG

    wd = str(working_dir or WORKING_DIR)
    os.makedirs(wd, exist_ok=True)
    rag = LightRAG(
        working_dir=wd,
        llm_model_func=_llm_model_func,
        embedding_func=_embedding_func(),
    )
    await rag.initialize_storages()  # inicializa storages + pipeline status
    return rag


def grafo_existe(working_dir=None):
    wd = Path(working_dir or WORKING_DIR)
    return (wd / "graph_chunk_entity_relation.graphml").exists()


def ler_graphml(caminho=None):
    """Le o GraphML do LightRAG de forma ROBUSTA (NetworkX).

    O arquivo as vezes vem com null bytes/lixo apos </graphml> (artefato de
    escrita/sandbox), o que quebra o parser XML. Aqui limpamos os null bytes e
    truncamos no fim do XML antes de parsear.
    """
    import networkx as nx

    caminho = Path(caminho or GRAPHML)
    if not caminho.exists():
        raise FileNotFoundError(
            f"Grafo nao encontrado em {caminho}. Rode antes: python 01_indexar_grafo.py")
    bruto = caminho.read_bytes().replace(b"\x00", b"")
    fim = bruto.rfind(b"</graphml>")
    if fim != -1:
        bruto = bruto[:fim + len(b"</graphml>")]
    return nx.parse_graphml(bruto.decode("utf-8", errors="ignore"))
