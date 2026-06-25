"""
_comum.py - Funcoes auxiliares compartilhadas pelos scripts da Aula 6.

Aula 6 = Indexacao Avancada (Parent-Child, RAPTOR, HyDE) com Haystack.
Stack: Haystack + Ollama (embeddings) + Groq (LLM) + OpenSearch (vetores) +
LangFuse (observabilidade no chat, script 06).

Corpus:
  - Por padrao usa o "corpus de trabalho" (datasets/corpus_trabalho.json) gerado
    pelo 01_gerar_dataset.py (amostra do TCU da Aula 4) + as perguntas geradas.
  - Se eles nao existirem, cai no corpus original (corpus_indexacao_avancada.json).

Voce NAO executa este arquivo diretamente. Ele e importado pelos outros scripts.
"""

import importlib.util
import json
import os
from pathlib import Path

from dotenv import load_dotenv

PASTA_SCRIPTS = Path(__file__).resolve().parent
PASTA_AULA6 = PASTA_SCRIPTS.parent
PASTA_PROJETO = PASTA_AULA6.parent
PASTA_DATASETS = PASTA_AULA6 / "datasets"

CORPUS_ORIGINAL = PASTA_DATASETS / "corpus_indexacao_avancada.json"
CORPUS_TRABALHO = PASTA_DATASETS / "corpus_trabalho.json"
PERGUNTAS_GERADAS = PASTA_DATASETS / "perguntas_geradas.json"
CORPUS_ACORDAOS_AULA4 = PASTA_PROJETO / "aula4" / "datasets" / "corpus_juridico_aula4_v2.json"


def langfuse_configurado():
    return bool(os.getenv("LANGFUSE_SECRET_KEY") and os.getenv("LANGFUSE_PUBLIC_KEY"))


def carregar_env():
    caminho = None
    for c in [PASTA_PROJETO / ".env", PASTA_AULA6 / ".env", PASTA_SCRIPTS / ".env"]:
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


def carregar_documentos():
    """Lista de documentos [{id, tipo, tribunal, texto}] (trabalho > original)."""
    if CORPUS_TRABALHO.exists():
        with open(CORPUS_TRABALHO, "r", encoding="utf-8") as f:
            return json.load(f)
    with open(CORPUS_ORIGINAL, "r", encoding="utf-8") as f:
        return json.load(f).get("documentos", [])


def carregar_perguntas():
    """Lista de perguntas (geradas > perguntas_teste do original)."""
    if PERGUNTAS_GERADAS.exists():
        with open(PERGUNTAS_GERADAS, "r", encoding="utf-8") as f:
            return json.load(f)
    with open(CORPUS_ORIGINAL, "r", encoding="utf-8") as f:
        return json.load(f).get("perguntas_teste", [])


def carregar_corpus():
    return carregar_documentos(), carregar_perguntas()


def carregar_acordaos_aula4(limite=0):
    """Le os acordaos do TCU (Aula 4). Cada doc ganha 'texto' = titulo + conteudo."""
    with open(CORPUS_ACORDAOS_AULA4, "r", encoding="utf-8") as f:
        docs = json.load(f)
    if limite and limite > 0:
        docs = docs[:limite]
    for d in docs:
        d["texto"] = f"{d.get('titulo', '')}. {d.get('conteudo', '')}".strip()
    return docs


def documentos_haystack():
    from haystack import Document

    return [
        Document(content=d.get("texto", ""),
                 meta={"id_original": d.get("id"), "tipo": d.get("tipo", ""),
                       "tribunal": d.get("tribunal", "")})
        for d in carregar_documentos()
    ]


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


def abrir_store(indice):
    from haystack_integrations.document_stores.opensearch import OpenSearchDocumentStore

    _, modelo = config_ollama()
    os_cfg = config_opensearch()
    auth = (os_cfg["usuario"], os_cfg["senha"]) if os_cfg["usuario"] else None
    return OpenSearchDocumentStore(
        hosts=os_cfg["url"], index=indice, embedding_dim=dimensao_do_modelo(modelo),
        http_auth=auth, use_ssl=False, verify_certs=False,
    )


def doc_embedder():
    from haystack_integrations.components.embedders.ollama import OllamaDocumentEmbedder

    base_url, modelo = config_ollama()
    return OllamaDocumentEmbedder(model=modelo, url=base_url)


def text_embedder():
    from haystack_integrations.components.embedders.ollama import OllamaTextEmbedder

    base_url, modelo = config_ollama()
    return OllamaTextEmbedder(model=modelo, url=base_url)


def groq_client():
    from openai import OpenAI

    api_key, modelo, base_url = config_groq()
    return OpenAI(api_key=api_key, base_url=base_url), modelo


def gerar_texto(cliente, modelo, prompt, max_tokens=500, temperature=0.2):
    resp = cliente.chat.completions.create(
        model=modelo, messages=[{"role": "user", "content": prompt}],
        temperature=temperature, max_tokens=max_tokens,
    )
    # Alguns modelos (ex.: de reasoning) ou completacoes filtradas devolvem
    # content = None/"" - protegemos para nao quebrar quem usa o texto depois.
    conteudo = resp.choices[0].message.content
    return (conteudo or "").strip()


PROMPT_RESPOSTA = (
    "Voce e um assistente juridico. Responda APENAS com base nos trechos abaixo, "
    "de forma objetiva. Se nao constar, diga que nao consta.\n\n"
    "Trechos:\n{contextos}\n\nPergunta: {pergunta}\nResposta:"
)


def responder_com_contexto(cliente, modelo, pergunta, contextos):
    bloco = "\n".join(f"- {c}" for c in contextos) if contextos else "(sem contexto)"
    return gerar_texto(cliente, modelo, PROMPT_RESPOSTA.format(contextos=bloco, pergunta=pergunta))


def importar_script(nome_arquivo):
    """Importa um script irmao (nome comeca com numero) como modulo. Usado pelo 05."""
    caminho = PASTA_SCRIPTS / nome_arquivo
    spec = importlib.util.spec_from_file_location(nome_arquivo.replace(".py", ""), caminho)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo
