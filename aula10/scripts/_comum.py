"""
_comum.py - Funcoes auxiliares compartilhadas pelos scripts da Aula 10.

Aula 10 = Agentic RAG + Adaptive RAG (padrao ReAct) com HAYSTACK.
Stack: Haystack (Agent/Tool/ConditionalRouter) + Groq (LLM com tool-calling) +
Ollama (embeddings) + OpenSearch (jurisprudencia, reusa o indice da Aula 4) +
SQLite (dados estruturados) + Tavily (web, opcional) + LangFuse (observabilidade).

As 3 FERRAMENTAS do agente:
  - buscar_documentos : busca semantica na jurisprudencia (OpenSearch/Ollama)
  - buscar_web        : fatos recentes na web (Tavily, com fallback offline)
  - consultar_banco   : dados estruturados via text-to-SQL no SQLite (so SELECT)

Voce NAO executa este arquivo diretamente. Ele e importado pelos outros scripts.
"""

import os
import re
import sqlite3
from pathlib import Path

from dotenv import load_dotenv

PASTA_SCRIPTS = Path(__file__).resolve().parent
PASTA_AULA10 = PASTA_SCRIPTS.parent
PASTA_PROJETO = PASTA_AULA10.parent
PASTA_DATASETS = PASTA_AULA10 / "datasets"

INDICE_TCU = os.getenv("AULA10_INDICE", "aula4_hibrido")          # reusa jurisprudencia da Aula 4
CORPUS_ACORDAOS_AULA4 = PASTA_PROJETO / "aula4" / "datasets" / "corpus_juridico_aula4_v2.json"
DB_PATH = PASTA_DATASETS / "juridico_segpub.db"                   # dados estruturados (SQLite)

DIMENSAO_EMBEDDING = {"nomic-embed-text": 768, "mxbai-embed-large": 1024, "bge-m3": 1024}


# ---------------------------------------------------------------------------
# Ambiente / configuracoes
# ---------------------------------------------------------------------------
def langfuse_configurado():
    return bool(os.getenv("LANGFUSE_SECRET_KEY") and os.getenv("LANGFUSE_PUBLIC_KEY"))


def tavily_configurado():
    return bool(os.getenv("TAVILY_API_KEY"))


def carregar_env():
    caminho = None
    for c in [PASTA_PROJETO / ".env", PASTA_AULA10 / ".env", PASTA_SCRIPTS / ".env"]:
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


def config_groq():
    """(api_key, modelo, base_url). LLM com TOOL-CALLING; nao usar reasoning aqui."""
    return (os.getenv("GROQ_API_KEY", ""),
            os.getenv("AULA10_LLM_MODEL", "llama-3.3-70b-versatile"),
            "https://api.groq.com/openai/v1")


def config_ollama():
    return (os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text"))


def config_opensearch():
    host = os.getenv("OPENSEARCH_HOST", "localhost")
    porta = os.getenv("OPENSEARCH_PORT", "9200")
    return {"url": f"http://{host}:{porta}",
            "usuario": os.getenv("OPENSEARCH_USER", ""), "senha": os.getenv("OPENSEARCH_PASS", "")}


def dimensao_do_modelo(nome_modelo):
    return DIMENSAO_EMBEDDING.get(nome_modelo.split(":")[0].lower(), 768)


# ---------------------------------------------------------------------------
# LLM (Groq) - chat generator (tool calling) e chamada simples
# ---------------------------------------------------------------------------
def chat_generator(temperatura=0.1, max_tokens=800):
    """OpenAIChatGenerator apontando para a Groq (suporta tool calling)."""
    from haystack.components.generators.chat import OpenAIChatGenerator
    from haystack.utils import Secret

    api_key, modelo, base_url = config_groq()
    return OpenAIChatGenerator(api_key=Secret.from_token(api_key), model=modelo,
                               api_base_url=base_url,
                               generation_kwargs={"temperature": temperatura, "max_tokens": max_tokens})


def groq_client():
    from openai import OpenAI

    api_key, modelo, base_url = config_groq()
    return OpenAI(api_key=api_key, base_url=base_url), modelo


def gerar_texto(cliente, modelo, prompt, max_tokens=400, temperature=0.0):
    resp = cliente.chat.completions.create(
        model=modelo, messages=[{"role": "user", "content": prompt}],
        temperature=temperature, max_tokens=max_tokens)
    return (resp.choices[0].message.content or "").strip()


# ---------------------------------------------------------------------------
# OpenSearch (busca densa) - reusa o indice do TCU
# ---------------------------------------------------------------------------
def abrir_store(indice):
    from haystack_integrations.document_stores.opensearch import OpenSearchDocumentStore

    _, modelo = config_ollama()
    os_cfg = config_opensearch()
    auth = (os_cfg["usuario"], os_cfg["senha"]) if os_cfg["usuario"] else None
    return OpenSearchDocumentStore(hosts=os_cfg["url"], index=indice,
                                   embedding_dim=dimensao_do_modelo(modelo),
                                   http_auth=auth, use_ssl=False, verify_certs=False)


def text_embedder():
    from haystack_integrations.components.embedders.ollama import OllamaTextEmbedder

    base_url, modelo = config_ollama()
    return OllamaTextEmbedder(model=modelo, url=base_url)


_PIPE_BUSCA = None  # cache do pipeline de busca densa (1 por processo)


def _pipe_busca(top_k=4):
    global _PIPE_BUSCA
    if _PIPE_BUSCA is None:
        from haystack import Pipeline
        from haystack_integrations.components.retrievers.opensearch import (
            OpenSearchEmbeddingRetriever,
        )

        store = abrir_store(INDICE_TCU)
        p = Pipeline()
        p.add_component("embedder", text_embedder())
        p.add_component("retriever", OpenSearchEmbeddingRetriever(document_store=store, top_k=top_k))
        p.connect("embedder.embedding", "retriever.query_embedding")
        _PIPE_BUSCA = p
    return _PIPE_BUSCA


# ---------------------------------------------------------------------------
# FERRAMENTA 1: buscar_documentos (jurisprudencia, OpenSearch)
# ---------------------------------------------------------------------------
def buscar_documentos(query: str) -> str:
    """Busca semantica na jurisprudencia local e devolve os trechos como texto."""
    try:
        docs = _pipe_busca().run({"embedder": {"text": query}})["retriever"]["documents"]
    except Exception as e:
        return f"[erro ao buscar documentos: {e}]"
    if not docs:
        return "Nenhum documento local relevante encontrado."
    linhas = []
    for i, d in enumerate(docs, 1):
        ident = d.meta.get("id_original", "?")
        linhas.append(f"[Doc {i} | {ident}] {d.content[:500]}")
    return "\n".join(linhas)


# ---------------------------------------------------------------------------
# FERRAMENTA 2: buscar_web (Tavily com fallback offline)
# ---------------------------------------------------------------------------
def buscar_web(query: str) -> str:
    """Busca na web (Tavily). Sem TAVILY_API_KEY, devolve aviso offline."""
    if not tavily_configurado():
        return ("[WEB SEARCH OFFLINE] Sem TAVILY_API_KEY - nao foi possivel buscar na web. "
                "Configure a chave para fatos recentes.")
    try:
        from haystack_integrations.components.websearch.tavily import TavilyWebSearch

        docs = TavilyWebSearch(top_k=3).run(query=query)["documents"]
        return "\n".join(f"[Web {i}] {d.content[:500]}" for i, d in enumerate(docs, 1)) or "Sem resultados na web."
    except Exception as e:
        return f"[web search falhou: {e}]"


# ---------------------------------------------------------------------------
# FERRAMENTA 3: consultar_banco (text-to-SQL no SQLite, so SELECT)
# ---------------------------------------------------------------------------
SCHEMA_SQL = """
Tabelas (SQLite) do banco juridico/seguranca publica:
  acordaos(id, numero, tribunal, data_julgamento, relator, tipo, ementa, resultado, pena_anos REAL, area_direito)
  ocorrencias(id, numero_bo, data, tipo_crime, municipio, estado, status, valor_envolvido_reais REAL, descricao)
  legislacao(id, numero, tipo, data_publicacao, ementa, artigos_principais, status_vigencia)
  doutrina(id, titulo, autores, publicacao, ano INTEGER, resumo, temas)
"""


def _gerar_sql(consulta_nl: str) -> str:
    cliente, modelo = groq_client()
    prompt = (
        "Voce converte perguntas em SQL para SQLite. Gere UMA consulta SELECT (somente "
        "leitura) para responder a pergunta, usando o schema abaixo. Responda APENAS o SQL, "
        "sem explicacao, sem ``` e sem ponto-e-virgula,\n"
        "Para todos os campos string nas clausulas WHERE sempre usar COLLATE NOCASE para ignrorar maiusculo ou minusculo;\n"
        f"{SCHEMA_SQL}\nPergunta: {consulta_nl}\nSQL:")
    sql = gerar_texto(cliente, modelo, prompt, max_tokens=200, temperature=0.0)
    sql = re.sub(r"```(?:sql)?", "", sql).strip().rstrip(";").strip()
    return sql


def consultar_banco(consulta: str) -> str:
    """Consulta os dados ESTRUTURADOS (SQLite) via text-to-SQL. So executa SELECT."""
    if not DB_PATH.exists():
        return f"[banco {DB_PATH.name} nao existe - rode: python 01_preparar_dados.py]"
    sql = _gerar_sql(consulta)
    if not sql.lower().lstrip().startswith("select"):
        return f"[consulta rejeitada - apenas SELECT e permitido] SQL gerado: {sql}"
    if " limit " not in sql.lower():
        sql += " LIMIT 20"
    try:
        con = sqlite3.connect(DB_PATH)
        con.row_factory = sqlite3.Row
        linhas = con.execute(sql).fetchall()
        con.close()
    except Exception as e:
        return f"[erro SQL: {e}] SQL: {sql}"
    if not linhas:
        return f"(sem resultados)  SQL: {sql}"
    cols = linhas[0].keys()
    out = [f"SQL: {sql}", " | ".join(cols)]
    for r in linhas[:20]:
        out.append(" | ".join(str(r[c])[:60] for c in cols))
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Ferramentas Haystack (Tool) e criacao do Agente
# ---------------------------------------------------------------------------
def ferramentas():
    """Lista de Tool do Haystack (nome + descricao + schema + funcao)."""
    from haystack.tools import Tool

    def _p(desc):
        return {"type": "object",
                "properties": {"query": {"type": "string", "description": desc}},
                "required": ["query"]}

    t_docs = Tool(
        name="buscar_documentos",
        description=("Busca jurisprudencia e acordaos no acervo local (busca vetorial). "
                     "Use para leis, decisoes, conceitos juridicos. NAO use para fatos "
                     "recentes/noticias (use buscar_web). Args: query (str) em portugues."),
        parameters=_p("termos de busca juridica em portugues"),
        function=buscar_documentos)

    t_web = Tool(
        name="buscar_web",
        description=("Busca informacoes ATUAIS na web (noticias, normas recentes). Use "
                     "quando a pergunta envolver fatos recentes que podem nao estar no "
                     "acervo. Args: query (str)."),
        parameters=_p("termos de busca na web"),
        function=buscar_web)

    t_sql = Tool(
        name="consultar_banco",
        description=("Consulta dados ESTRUTURADOS (acordaos, ocorrencias, legislacao, "
                     "doutrina) por filtros, contagens, somas, ordenacoes (ex.: 'quantas "
                     "ocorrencias de estelionato em SP', 'acordaos com pena maior que 5 "
                     "anos'). Use para numeros/agregacoes. Args: query (str) em linguagem natural."),
        parameters={"type": "object",
                    "properties": {"consulta": {"type": "string",
                                                "description": "pergunta em linguagem natural sobre os dados"}},
                    "required": ["consulta"]},
        function=consultar_banco)
    return [t_docs, t_web, t_sql]


SYSTEM_AGENTE = (
    "Voce e um assistente juridico (controle externo, Direito e Seguranca Publica). "
    "Use as ferramentas quando precisar de informacao que nao domina. Pense passo a passo: "
    "decida a ferramenta, observe o resultado e repita se necessario. Responda em portugues, "
    "de forma objetiva, citando as fontes (id do documento, tabela do banco ou web). "
    "Se ja sabe a resposta (conhecimento geral), responda sem usar ferramentas. "
    "FALLBACK: se buscar_documentos ou consultar_banco NAO trouxerem informacao suficiente, "
    "tente buscar_web antes de concluir. Se ainda assim nao houver base, diga claramente que "
    "nao consta nas fontes consultadas."
)


def criar_agente(max_passos=5, ferramentas_sel=None):
    """Cria o Agent do Haystack (laco ReAct/tool-calling) com as 3 ferramentas."""
    from haystack.components.agents import Agent

    return Agent(
        chat_generator=chat_generator(),
        tools=ferramentas_sel if ferramentas_sel is not None else ferramentas(),
        system_prompt=SYSTEM_AGENTE,
        exit_conditions=["text"],     # encerra quando o LLM responde texto (sem tool call)
        max_agent_steps=max_passos,   # guard contra loop infinito
    )
