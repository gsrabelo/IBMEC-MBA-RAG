"""
_comum.py - Funcoes auxiliares compartilhadas pelos scripts da Aula 8.

Aula 8 = RAG Reflexivo e Auto-Corretivo: Self-RAG, CRAG e roteamento condicional.
Stack: Haystack + Ollama (embeddings) + Groq (LLM/avaliador) + OpenSearch (vetores)
       + Tavily (web search de fallback, opcional) + LangFuse (observabilidade).

IMPORTANTE (decisoes desta aula):
  - Orquestracao do CRAG via Haystack **ConditionalRouter** (sem LangGraph).
  - Web search com Tavily SE houver TAVILY_API_KEY no .env; senao, fallback OFFLINE
    (stub) para os scripts rodarem sempre.
  - Reaproveita o indice do TCU da Aula 4 ('aula4_hibrido').

Voce NAO executa este arquivo diretamente. Ele e importado pelos outros scripts.
"""

import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv

PASTA_SCRIPTS = Path(__file__).resolve().parent
PASTA_AULA8 = PASTA_SCRIPTS.parent
PASTA_PROJETO = PASTA_AULA8.parent
PASTA_DATASETS = PASTA_AULA8 / "datasets"

# Reaproveita o indice de acordaos do TCU (mesmo da Aula 4/6/7).
INDICE_TCU = os.getenv("AULA8_INDICE", "aula4_hibrido")
CORPUS_ACORDAOS_AULA4 = PASTA_PROJETO / "aula4" / "datasets" / "corpus_juridico_aula4_v2.json"
# Queries de teste calibradas (algumas forcam o web search).
QUERIES_TESTE = PASTA_DATASETS / "queries_teste_aula8.json"
QUERIES_CRAG = PASTA_DATASETS / "queries_crag_tcu.json"  # curado p/ o TCU (local vs web)


# ---------------------------------------------------------------------------
# Ambiente / .env
# ---------------------------------------------------------------------------
def langfuse_configurado():
    return bool(os.getenv("LANGFUSE_SECRET_KEY") and os.getenv("LANGFUSE_PUBLIC_KEY"))


def tavily_configurado():
    return bool(os.getenv("TAVILY_API_KEY"))


def carregar_env():
    caminho = None
    for c in [PASTA_PROJETO / ".env", PASTA_AULA8 / ".env", PASTA_SCRIPTS / ".env"]:
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


# ---------------------------------------------------------------------------
# Configuracoes
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
# Corpus / OpenSearch / embeddings
# ---------------------------------------------------------------------------
def carregar_acordaos_aula4(limite=0):
    """Le os acordaos do TCU (Aula 4). Cada doc ganha 'texto' = titulo + conteudo."""
    with open(CORPUS_ACORDAOS_AULA4, "r", encoding="utf-8") as f:
        docs = json.load(f)
    if limite and limite > 0:
        docs = docs[:limite]
    for d in docs:
        d["texto"] = f"{d.get('titulo', '')}. {d.get('conteudo', '')}".strip()
    return docs


def documentos_haystack(limite=0):
    """Acordaos do TCU como Documents do Haystack (para o 01 indexar)."""
    from haystack import Document

    return [
        Document(content=d["texto"],
                 meta={"id_original": d["id"], "tipo": d.get("tipo", "acordao")})
        for d in carregar_acordaos_aula4(limite=limite)
    ]


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


def montar_busca(store, top_k):
    """Pipeline simples de busca densa: embedder (Ollama) -> retriever (OpenSearch)."""
    from haystack import Pipeline
    from haystack_integrations.components.retrievers.opensearch import (
        OpenSearchEmbeddingRetriever,
    )

    pipe = Pipeline()
    pipe.add_component("embedder", text_embedder())
    pipe.add_component("retriever", OpenSearchEmbeddingRetriever(document_store=store, top_k=top_k))
    pipe.connect("embedder.embedding", "retriever.query_embedding")
    return pipe


def buscar(pipe, query):
    """Roda a busca densa para uma query e devolve a lista de Documents."""
    return pipe.run({"embedder": {"text": query}})["retriever"]["documents"]


# ---------------------------------------------------------------------------
# Groq (LLM) - chamadas diretas e helpers
# ---------------------------------------------------------------------------
def groq_client():
    from openai import OpenAI

    api_key, modelo, base_url = config_groq()
    return OpenAI(api_key=api_key, base_url=base_url), modelo


def _eh_reasoning(modelo):
    """Heuristica: o modelo e de 'raciocinio' (gasta tokens pensando)?"""
    m = (modelo or "").lower()
    return any(t in m for t in ["gpt-oss", "deepseek-r1", "qwq", "o1", "o3", "reason"])


def gerar_texto(cliente, modelo, prompt, max_tokens=400, temperature=0.2):
    """Chama o LLM e devolve o texto, robusto a modelos de RACIOCINIO.

    Modelos de reasoning (ex.: gpt-oss) consomem o orcamento de tokens 'pensando' e,
    com max_tokens baixo, devolvem content VAZIO. Aqui: pedimos reasoning_effort baixo,
    damos folga de tokens, protegemos content None/"" e removemos blocos <think>.
    """
    kwargs = dict(model=modelo, messages=[{"role": "user", "content": prompt}],
                  temperature=temperature, max_tokens=max_tokens)
    if _eh_reasoning(modelo):
        kwargs["max_tokens"] = max(max_tokens, 1024)
        kwargs["extra_body"] = {"reasoning_effort": "low"}
    try:
        resp = cliente.chat.completions.create(**kwargs)
    except Exception:
        kwargs.pop("extra_body", None)  # SDK/endpoint sem suporte a reasoning_effort
        resp = cliente.chat.completions.create(**kwargs)
    conteudo = (resp.choices[0].message.content or "").strip()
    return re.sub(r"<think>.*?</think>", "", conteudo, flags=re.DOTALL).strip()


def gen_kwargs(modelo, temperature, max_tokens):
    """generation_kwargs para o OpenAIGenerator (Haystack), robusto a reasoning.

    Os componentes Haystack (Self-RAG/CRAG) usam OpenAIGenerator, que NAO passa pelo
    gerar_texto. Para modelos de raciocinio, pedimos reasoning_effort baixo e damos
    folga de tokens - senao a saida pode vir vazia (mesmo problema do gerar_texto).
    """
    kw = {"temperature": temperature, "max_tokens": max_tokens}
    if _eh_reasoning(modelo):
        kw["max_tokens"] = max(max_tokens, 1024)
        kw["reasoning_effort"] = "low"
    return kw


def extrair_json(texto):
    """Extrai o 1o objeto JSON de um texto (LLM as vezes adiciona comentarios)."""
    import re

    m = re.search(r"\{.*\}", texto, re.DOTALL)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return {}


# ---------------------------------------------------------------------------
# Avaliador de relevancia (LLM-as-Judge) - coracao do CRAG
# ---------------------------------------------------------------------------
PROMPT_AVALIADOR = (
    "Voce e um avaliador de relevancia de documentos para sistemas RAG juridicos.\n\n"
    "Pergunta: {pergunta}\n"
    "Documento: {documento}\n\n"
    "Avalie o quanto este documento ajuda a responder a pergunta.\n"
    "Responda APENAS um JSON: {{\"score\": <0 a 1>, \"motivo\": \"<curto>\"}}\n"
    "- 1.0 = altamente relevante, responde diretamente\n"
    "- 0.5 = parcialmente relevante, ajuda mas incompleto\n"
    "- 0.0 = irrelevante\n"
)


def avaliar_documento(cliente, modelo, pergunta, documento):
    """Pede ao LLM um score 0-1 de relevancia de UM documento para a pergunta."""
    prompt = PROMPT_AVALIADOR.format(pergunta=pergunta, documento=documento[:1500])
    dados = extrair_json(gerar_texto(cliente, modelo, prompt, max_tokens=120, temperature=0.0))
    try:
        score = float(dados.get("score", 0.0))
    except (TypeError, ValueError):
        score = 0.0
    return max(0.0, min(1.0, score)), dados.get("motivo", "")


def avaliar_documentos(cliente, modelo, pergunta, documentos):
    """Avalia uma lista de Documents; devolve (score_medio, lista_de_(doc, score, motivo))."""
    detalhes = []
    for d in documentos:
        score, motivo = avaliar_documento(cliente, modelo, pergunta, d.content)
        detalhes.append((d, score, motivo))
    media = sum(s for _, s, _ in detalhes) / len(detalhes) if detalhes else 0.0
    return media, detalhes


# ---------------------------------------------------------------------------
# Roteamento do CRAG (limiares)
# ---------------------------------------------------------------------------
LIMITE_ALTO = float(os.getenv("CRAG_LIMITE_ALTO", "0.7"))   # >= alto -> so local
LIMITE_BAIXO = float(os.getenv("CRAG_LIMITE_BAIXO", "0.3"))  # < baixo -> so web


def decidir_rota(score):
    """Mapeia o score do avaliador para a rota CRAG: 'local' | 'fusao' | 'web'."""
    if score >= LIMITE_ALTO:
        return "local"
    if score >= LIMITE_BAIXO:
        return "fusao"
    return "web"


# ---------------------------------------------------------------------------
# Web search (Tavily com fallback OFFLINE)
# ---------------------------------------------------------------------------
def web_search(query, max_results=3):
    """Busca na web via Tavily se houver chave; senao devolve stub offline.

    Retorna lista de dicts: {"url", "content", "score"}.
    """
    if not tavily_configurado():
        return [{
            "url": "",
            "content": ("[WEB SEARCH OFFLINE] Sem TAVILY_API_KEY no .env - o fallback "
                        "de web search nao foi executado. Configure a chave para buscar "
                        "informacoes externas/atualizadas."),
            "score": 0.0,
        }]
    try:
        from tavily import TavilyClient

        cliente = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
        resp = cliente.search(query=query, max_results=max_results, search_depth="advanced")
        return [{"url": r.get("url", ""), "content": r.get("content", ""),
                 "score": r.get("score", 0.0)} for r in resp.get("results", [])]
    except Exception as e:  # rede/lib/chave invalida -> nao derruba o pipeline
        return [{"url": "", "content": f"[WEB SEARCH FALHOU] {e}", "score": 0.0}]


def web_para_documents(resultados):
    """Converte resultados de web_search em Documents do Haystack."""
    from haystack import Document

    return [
        Document(content=r["content"],
                 meta={"id_original": r.get("url") or "web", "tipo": "web", "score_web": r.get("score", 0.0)})
        for r in resultados if r.get("content")
    ]


# ---------------------------------------------------------------------------
# Geracao da resposta final (com contexto)
# ---------------------------------------------------------------------------
PROMPT_RESPOSTA = (
    "Voce e um assistente juridico especializado em controle externo (TCU). "
    "Responda APENAS com base nos trechos abaixo, de forma objetiva. "
    "Se a informacao vier da web, deixe claro. Se nao constar, diga que nao consta.\n\n"
    "Trechos:\n{contextos}\n\nPergunta: {pergunta}\nResposta:"
)


def responder_com_contexto(cliente, modelo, pergunta, documentos):
    """Gera a resposta final a partir de uma lista de Documents (locais e/ou web)."""
    linhas = []
    for d in documentos:
        origem = "WEB" if d.meta.get("tipo") == "web" else "LOCAL"
        linhas.append(f"[{origem}] {d.content}")
    bloco = "\n".join(linhas) if linhas else "(sem contexto)"
    return gerar_texto(cliente, modelo,
                       PROMPT_RESPOSTA.format(contextos=bloco, pergunta=pergunta),
                       max_tokens=500, temperature=0.2)


# ---------------------------------------------------------------------------
# Self-RAG (training-free) - tokens de controle emitidos por prompting
# ---------------------------------------------------------------------------
PROMPT_RETRIEVE = (
    "Decida se responder a pergunta abaixo exige consultar documentos especificos "
    "(jurisprudencia, leis, acordaos) ou se e conhecimento geral.\n"
    "Responda APENAS JSON: {{\"retrieve\": \"yes\"|\"no\", \"motivo\": \"<curto>\"}}\n\n"
    "Pergunta: {pergunta}"
)
PROMPT_ISSUP = (
    "Verifique se a RESPOSTA abaixo tem suporte factual nos TRECHOS fornecidos.\n"
    "Responda APENAS JSON: {{\"issup\": \"fully\"|\"partially\"|\"no\", \"motivo\": \"<curto>\"}}\n\n"
    "Trechos:\n{contextos}\n\nResposta: {resposta}"
)
PROMPT_ISUSE = (
    "De uma nota de utilidade (1 a 5) para a RESPOSTA em relacao a PERGUNTA.\n"
    "Responda APENAS JSON: {{\"isuse\": <1-5>, \"motivo\": \"<curto>\"}}\n\n"
    "Pergunta: {pergunta}\nResposta: {resposta}"
)


def token_retrieve(cliente, modelo, pergunta):
    dados = extrair_json(gerar_texto(cliente, modelo,
                                     PROMPT_RETRIEVE.format(pergunta=pergunta),
                                     max_tokens=80, temperature=0.0))
    return dados.get("retrieve", "yes").lower(), dados.get("motivo", "")


def token_issup(cliente, modelo, contextos, resposta):
    bloco = "\n".join(f"- {c}" for c in contextos) if contextos else "(sem contexto)"
    dados = extrair_json(gerar_texto(cliente, modelo,
                                     PROMPT_ISSUP.format(contextos=bloco, resposta=resposta),
                                     max_tokens=80, temperature=0.0))
    return dados.get("issup", "no").lower(), dados.get("motivo", "")


def token_isuse(cliente, modelo, pergunta, resposta):
    dados = extrair_json(gerar_texto(cliente, modelo,
                                     PROMPT_ISUSE.format(pergunta=pergunta, resposta=resposta),
                                     max_tokens=80, temperature=0.0))
    try:
        nota = int(float(dados.get("isuse", 3)))
    except (TypeError, ValueError):
        nota = 3
    return max(1, min(5, nota)), dados.get("motivo", "")


def importar_script(nome_arquivo):
    """Importa um script irmao (nome comeca com numero) como modulo."""
    import importlib.util

    caminho = PASTA_SCRIPTS / nome_arquivo
    spec = importlib.util.spec_from_file_location(nome_arquivo.replace(".py", ""), caminho)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


def carregar_queries_teste():
    """Carrega as queries curadas (local vs web) adaptadas ao indice do TCU.

    Formato: lista de {"id","pergunta","rota_esperada":"local"|"web","motivo"}.
    Se o arquivo nao existir, devolve lista vazia (os scripts seguem sem demo guiado).
    """
    if not QUERIES_CRAG.exists():
        return []
    with open(QUERIES_CRAG, "r", encoding="utf-8") as f:
        return json.load(f)


def criar_tavily(top_k=3):
    """Cria o componente OFICIAL TavilyWebSearch (Haystack) ou None se nao der.

    Usa TAVILY_API_KEY do ambiente. Se a chave nao existir ou o pacote
    tavily-haystack nao estiver instalado, devolve None (o chamador cai no
    fallback offline). Assim os scripts rodam sempre.
    """
    if not tavily_configurado():
        return None
    try:
        from haystack_integrations.components.websearch.tavily import TavilyWebSearch

        return TavilyWebSearch(top_k=top_k, search_params={"search_depth": "advanced"})
    except Exception:
        return None
