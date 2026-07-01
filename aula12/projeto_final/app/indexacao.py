"""
indexacao.py - Decide ESTRATEGIA de indexacao (destino + chunking) e grava no storage.

Decisoes TRANSPARENTES (heuristicas, sem LLM):
  DESTINO  : 'grafo' (LightRAG) se texto longo e rico em entidades (multi-hop);
             senao 'opensearch'. Override: 'auto' | 'opensearch' | 'grafo'.

  CHUNKING (so quando destino=opensearch) - escolhe a melhor das tecnicas:
     - tabela          : documento tem tabelas -> 1 chunk por LINHA-FOLHA da tabela,
                         prefixado com a cadeia de ancestrais pelo codigo (estrutura-aware,
                         ver chunkar_tabela_hierarquica). Paragrafos fora de tabela (Notas)
                         viram chunks proprios presos ao titulo vigente.
     - hierarquico     : documento estruturado em secoes (varios titulos) - split GENERICO
                         por contagem de palavras (nao e' ciente da tabela em si).
     - sentenca_janela : texto de lei / denso em artigos -> precisao por sentenca
     - semantico       : texto longo e heterogeneo -> corta em mudancas de topico
     - recursivo       : texto corrido (default robusto: respeita paragrafos/sentencas)
     - fixo            : documento curto / baseline
   Override: 'auto' | fixo | recursivo | sentenca_janela | semantico | hierarquico | tabela.

Cada tecnica usa um componente NATIVO do Haystack. Embedding: Ollama (nomic-embed-text).
"""

import asyncio
import re
from functools import partial

from haystack import Document

from . import config
from .log import obter_logger

log = obter_logger(__name__)

LIMIAR_ENTIDADES = 30
MIN_PALAVRAS_GRAFO = 800
TECNICAS_CHUNK = {"fixo", "recursivo", "sentenca_janela", "semantico", "hierarquico", "tabela"}

# ---------------------------------------------------------------------------
# Chunking ESTRUTURA-AWARE p/ tabelas hierarquicas em Markdown (ex.: NCM, leis,
# classificacoes) - ver chunkar_tabela_hierarquica() mais abaixo.
# ---------------------------------------------------------------------------
_RE_LINHA_TABELA = re.compile(r"^\s*\|(.*)\|\s*$")
_RE_SEPARADOR = re.compile(r"^[\s:|-]+$")
_RE_NCM = re.compile(r"^\d{2,4}(\.\d{1,2}){0,3}$")
_RE_ALIQUOTA = re.compile(r"^(NT|\d+(,\d+)?)$")
_RE_HEADING = re.compile(r"^#{1,4}\s+(.*)")
_RE_NAO_DIGITO = re.compile(r"\D")


def _celulas_tabela(linha):
    m = _RE_LINHA_TABELA.match(linha)
    return None if not m else [c.strip() for c in m.group(1).split("|")]


def _codigo_normalizado(codigo):
    """Remove os pontos do codigo NCM p/ comparar por PREFIXO numerico puro.

    Necessario porque a profundidade de '-'/'--' na descricao NAO e' confiavel:
    p.ex. '0207.14.1 Pedacos nao desossados' nao tem traco algum, mesmo sendo mais
    profundo que '0207.14 -- Pedacos e miudezas, congelados'. O codigo, sem pontos,
    e' monotonico: '0207' -> '02071' -> '020714' -> '0207141' -> '02071411'."""
    return _RE_NAO_DIGITO.sub("", codigo or "")


def chunkar_tabela_hierarquica(conteudo):
    """1 chunk por LINHA-FOLHA da tabela (a que tem valor na ultima coluna), prefixado
    com o titulo da secao/capitulo vigente + a cadeia de ancestrais (linhas-categoria
    acima na hierarquia do CODIGO NCM). Sem isso, uma linha como 'Peitos | 0' fica orfa
    do contexto 'aves da especie Gallus domesticus' que da sentido a ela - e uma busca
    por sinonimos ('frango') nunca encontra o chunk certo.

    Paragrafos fora de tabela (ex.: 'Notas.' de Secao/Capitulo) viram chunks proprios,
    presos ao titulo vigente - o prompt de RAG depende de poder citar essas notas.
    """
    titulo_atual = ""
    ancestrais = []  # [(codigo_normalizado, rotulo)], do mais GERAL pro mais ESPECIFICO
    buffer_prosa = []
    docs = []

    def _flush_prosa():
        texto = " ".join(buffer_prosa).strip()
        buffer_prosa.clear()
        if texto:
            docs.append(Document(content=f"{titulo_atual}: {texto}" if titulo_atual else texto))

    for linha in (conteudo or "").splitlines():
        m_tit = _RE_HEADING.match(linha)
        if m_tit:
            _flush_prosa()
            titulo_atual = m_tit.group(1).strip()
            ancestrais = []
            continue

        celulas = _celulas_tabela(linha)
        if celulas is None:
            if linha.strip():
                buffer_prosa.append(linha.strip())
            continue
        _flush_prosa()
        if all(_RE_SEPARADOR.match(c or "-") for c in celulas):
            continue  # linha separadora |---|---|---|
        celulas_nv = [c for c in celulas if c]
        if not celulas_nv:
            continue
        if (celulas_nv[0] or "").upper() == "NCM" or "ALÍQUOTA" in (celulas_nv[-1] or "").upper():
            continue  # linha de cabecalho da tabela (repete a cada pagina/capitulo)

        codigo = celulas_nv[0] if _RE_NCM.match(celulas_nv[0] or "") else None
        ultima = celulas_nv[-1]
        tem_aliquota = bool(_RE_ALIQUOTA.match(ultima))
        descricao = (celulas_nv[-2] if tem_aliquota and len(celulas_nv) >= 2
                    else " ".join(celulas_nv[1:] if codigo else celulas_nv))
        descricao = (descricao or "").strip()

        # Defesa contra celulas desalinhadas pelo Docling (tabelas densas/linhas proximas
        # podem mesclar 2 linhas numa so, ou deixar a descricao vazia/igual ao codigo/igual
        # a propria aliquota). Preferimos PERDER a linha a indexar um chunk confiante com
        # codigo/descricao errados, que engana o LLM numa resposta de alta confianca.
        suspeita = (not descricao or (codigo and descricao == codigo)
                   or bool(_RE_ALIQUOTA.match(descricao)))
        if suspeita:
            log.debug("Linha suspeita (celula desalinhada) descartada: %r", linha.strip())
            continue

        rotulo = f"{codigo + ' ' if codigo else ''}{descricao}".strip()

        if codigo:
            cod_norm = _codigo_normalizado(codigo)
            # mantem so os ancestrais cujo codigo e' PREFIXO ESTRITO do atual (descarta
            # "tios"/"primos" de ramos ja encerrados, ex.: ao trocar de 02.07 p/ 02.06)
            ancestrais = [(c, r) for c, r in ancestrais if cod_norm.startswith(c) and cod_norm != c]
        else:
            cod_norm = None

        if not tem_aliquota:
            # linha-categoria (sem aliquota): vira ancestral p/ as linhas mais profundas
            if codigo:
                ancestrais.append((cod_norm, rotulo))
            continue

        # linha-folha: tem aliquota -> gera o chunk com a cadeia completa de ancestrais.
        # A info ESPECIFICA da linha vem PRIMEIRO, o contexto de ancestrais depois: itens
        # "irmaos" (ex.: 02.07 frango/peru/pato/ganso) compartilham um prefixo de contexto
        # longo e identico, que domina o embedding e afoga o que realmente diferencia cada
        # linha se vier na frente. Colocando o especifico primeiro, o termo que distingue
        # ("Peitos", "Gallus domesticus" via ancestral citado, "desossados") pesa mais.
        cadeia = [r for _, r in ancestrais]
        contexto = " > ".join(p for p in [titulo_atual] + cadeia if p)
        cabecalho = f"{rotulo} — Alíquota: {ultima}."
        texto = f"{cabecalho} Contexto: {contexto}" if contexto else cabecalho
        docs.append(Document(content=texto, meta={"ncm": codigo} if codigo else {}))

    _flush_prosa()
    return docs


# ---------------------------------------------------------------------------
# DESTINO (OpenSearch vs LightRAG)
# ---------------------------------------------------------------------------
def _entidades_distintas(conteudo):
    palavras = conteudo.split()
    if len(palavras) < MIN_PALAVRAS_GRAFO:
        return 0
    caps = {w.strip(".,;:()") for w in palavras if w[:1].isupper() and len(w) > 2}
    return len(caps)


def decidir_destino(dados, override="auto"):
    if override in ("opensearch", "grafo"):
        log.info("Destino forcado pelo usuario: %s", override)
        return override, f"forcado pelo usuario (override={override})"
    n_ent = _entidades_distintas(dados.get("conteudo", ""))
    log.debug("Heuristica de destino: %d entidades distintas (limiar=%d)", n_ent, LIMIAR_ENTIDADES)
    if n_ent >= LIMIAR_ENTIDADES:
        return "grafo", f"texto longo e rico em entidades ({n_ent} distintas >= {LIMIAR_ENTIDADES}) -> multi-hop"
    return "opensearch", f"texto/tabela direto ({n_ent} entidades distintas < {LIMIAR_ENTIDADES})"


# ---------------------------------------------------------------------------
# AVALIADOR de chunking (so para OpenSearch)
# ---------------------------------------------------------------------------
def _n_titulos(c):
    return len([l for l in c.splitlines() if l.lstrip().startswith("#")])


def _n_artigos(c):
    return len(re.findall(r"\bArt\.?\s*\d+", c))


def avaliar_chunking(dados, override="auto"):
    """Escolhe a melhor tecnica de chunking pela ESTRUTURA do documento (explicavel)."""
    conteudo = dados.get("conteudo", "")
    if override in TECNICAS_CHUNK:
        return override, f"forcado pelo usuario (chunking={override})"
    if dados.get("tabelas"):
        return "tabela", "documento tem tabelas -> 1 chunk por linha-folha (estrutura-aware)"

    n_pal, n_tit, n_art = len(conteudo.split()), _n_titulos(conteudo), _n_artigos(conteudo)
    log.debug("Sinais de chunking: %d palavras, %d titulos, %d 'Art.'", n_pal, n_tit, n_art)
    if n_tit >= 3:
        return "hierarquico", f"documento estruturado em secoes ({n_tit} titulos) -> hierarquico"
    if n_art >= 5:
        return "sentenca_janela", f"texto de lei/denso em artigos ({n_art} 'Art.') -> precisao por sentenca"
    if n_pal >= 1500:
        return "semantico", f"texto longo e heterogeneo ({n_pal} palavras) -> cortes por mudanca de topico"
    if n_pal >= 300:
        return "recursivo", f"texto corrido ({n_pal} palavras) -> recursivo (respeita paragrafos/sentencas)"
    return "fixo", f"documento curto ({n_pal} palavras) -> fixo (baseline)"


# ---------------------------------------------------------------------------
# Chunkers (componentes NATIVOS do Haystack)
# ---------------------------------------------------------------------------
def _rodar(splitter, conteudo):
    if hasattr(splitter, "warm_up"):
        splitter.warm_up()
    return splitter.run(documents=[Document(content=conteudo)])["documents"]


def _ollama_doc_embedder():
    from haystack_integrations.components.embedders.ollama import OllamaDocumentEmbedder
    base_url, modelo = config.config_ollama()
    return OllamaDocumentEmbedder(model=modelo, url=base_url)


def chunkar(conteudo, tecnica):
    from haystack.components.preprocessors import (DocumentSplitter,
        EmbeddingBasedDocumentSplitter, HierarchicalDocumentSplitter,
        RecursiveDocumentSplitter)

    log.debug("Chunkando com tecnica '%s' (%d caracteres)", tecnica, len(conteudo))
    if tecnica == "fixo":
        docs = _rodar(DocumentSplitter(split_by="word", split_length=200, split_overlap=0), conteudo)
    elif tecnica == "recursivo":
        docs = _rodar(RecursiveDocumentSplitter(split_length=200, split_overlap=30, split_unit="word"), conteudo)
    elif tecnica == "sentenca_janela":
        # janela = grupos de sentencas com sobreposicao (sentence-window)
        docs = _rodar(DocumentSplitter(split_by="sentence", split_length=3, split_overlap=1), conteudo)
    elif tecnica == "semantico":
        sp = EmbeddingBasedDocumentSplitter(document_embedder=_ollama_doc_embedder(),
                                            sentences_per_group=3, language="pt")
        docs = _rodar(sp, conteudo)
    elif tecnica == "hierarquico":
        nos = _rodar(HierarchicalDocumentSplitter(block_sizes={400, 100}, split_by="word"), conteudo)
        # indexa apenas as FOLHAS (chunks pequenos) - estrutura-aware
        docs = [d for d in nos if not d.meta.get("__children_ids")] or nos
    elif tecnica == "tabela":
        docs = chunkar_tabela_hierarquica(conteudo)
        if not docs:
            log.warning("Tecnica 'tabela': nenhuma linha-folha reconhecida no Markdown "
                       "-> fallback p/ documento inteiro (1 chunk).")
            docs = [Document(content=conteudo)]
    else:
        # tecnica desconhecida -> doc inteiro
        docs = [Document(content=conteudo)]

    # Descarta chunks vazios/so-espaco: embeddings de texto vazio sao degenerados
    # (vetor quase nulo) e o OpenSearch costuma atribuir a eles um score de
    # similaridade "neutro" que pode vencer conteudo relevante no top_k, fazendo
    # a mesma duvida de chunks vazios aparecer pra QUALQUER pergunta.
    antes = len(docs)
    docs = [d for d in docs if (d.content or "").strip()]
    if len(docs) < antes:
        log.warning("Tecnica '%s': descartados %d chunk(s) vazios/so-espaco (de %d gerados).",
                   tecnica, antes - len(docs), antes)
    return docs


# ---------------------------------------------------------------------------
# Gravacao: OpenSearch
# ---------------------------------------------------------------------------
def _store_opensearch():
    from haystack_integrations.document_stores.opensearch import OpenSearchDocumentStore
    os_cfg = config.config_opensearch()
    auth = (os_cfg["usuario"], os_cfg["senha"]) if os_cfg["usuario"] else None
    return OpenSearchDocumentStore(hosts=os_cfg["url"], index=os_cfg["indice"],
                                   embedding_dim=config.dimensao_embedding(),
                                   http_auth=auth, use_ssl=False, verify_certs=False)


def indexar_opensearch(docs, meta):
    from haystack.document_stores.types import DuplicatePolicy

    store = _store_opensearch()

    # Reingestao do mesmo arquivo: remove chunks antigos antes de gravar os novos.
    # Sem isso, (a) reingerir o MESMO conteudo gera o mesmo id (hash) e o write
    # falha com 409 version_conflict (a acao default e' 'create'); (b) reingerir
    # com OUTRA tecnica de chunking gera ids diferentes e os chunks antigos ficam
    # orfaos no indice, duplicando contexto nas buscas.
    arquivo = meta.get("arquivo")
    if arquivo:
        antigos = store.filter_documents(
            filters={"field": "meta.arquivo", "operator": "==", "value": arquivo})
        if antigos:
            store.delete_documents([d.id for d in antigos])
            log.info("Removidos %d chunk(s) antigos de '%s' antes de reindexar.",
                     len(antigos), arquivo)

    for d in docs:
        d.meta.update(meta)
    log.info("Gerando embeddings (Ollama) para %d chunk(s)...", len(docs))
    embedder = _ollama_doc_embedder()
    if hasattr(embedder, "warm_up"):
        embedder.warm_up()
    docs_emb = embedder.run(documents=docs)["documents"]
    log.info("Gravando %d documento(s) no OpenSearch (indice '%s')...",
             len(docs_emb), config.config_opensearch()["indice"])
    # OVERWRITE como rede de seguranca: mesmo que dois ids coincidam (ex.: dois
    # arquivos diferentes com um chunk identico), grava por cima em vez de falhar.
    store.write_documents(docs_emb, policy=DuplicatePolicy.OVERWRITE)
    return len(docs_emb)


# ---------------------------------------------------------------------------
# Gravacao: LightRAG (grafo)
# ---------------------------------------------------------------------------
async def _criar_lightrag():
    from lightrag import LightRAG
    from lightrag.llm.ollama import ollama_embed
    from lightrag.llm.openai import openai_complete_if_cache
    from lightrag.utils import EmbeddingFunc

    api_key, modelo, base_url = config.config_llm()
    o_base, o_modelo = config.config_ollama()

    async def llm_func(prompt, system_prompt=None, history_messages=None, **kwargs):
        return await openai_complete_if_cache(modelo, prompt, system_prompt=system_prompt,
                                              history_messages=history_messages or [],
                                              api_key=api_key, base_url=base_url, **kwargs)

    rag = LightRAG(working_dir=str(config.PASTA_RAG_STORAGE), llm_model_func=llm_func,
                   embedding_func=EmbeddingFunc(embedding_dim=config.dimensao_embedding(),
                       max_token_size=8192,
                       func=partial(ollama_embed.func, embed_model=o_modelo, host=o_base)))
    await rag.initialize_storages()
    return rag


def rodar_async(coro_factory):
    """Roda uma corrotina com seguranca, HAJA ou NAO um event loop ativo.

    O LightRAG e assincrono. Chamar asyncio.run() dentro de um endpoint 'async def'
    do FastAPI quebra ('asyncio.run() cannot be called from a running event loop').
    Aqui: se nao ha loop, usa asyncio.run direto; se ja ha um loop rodando, executa
    numa thread separada (que tem o proprio loop).
    """
    import concurrent.futures

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro_factory())  # sem loop ativo (caso comum)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        return ex.submit(lambda: asyncio.run(coro_factory())).result()


def indexar_grafo(conteudo):
    async def _run():
        log.info("Construindo grafo no LightRAG (varias chamadas de LLM, pode demorar)...")
        rag = await _criar_lightrag()
        try:
            await rag.ainsert(conteudo)
        finally:
            await rag.finalize_storages()
    rodar_async(_run)
    log.info("Grafo atualizado no LightRAG (storage: %s)", config.PASTA_RAG_STORAGE)
    return 1


# ---------------------------------------------------------------------------
# Orquestra a indexacao
# ---------------------------------------------------------------------------
def indexar(dados, meta, destino_override="auto", chunking_override="auto"):
    destino, motivo_destino = decidir_destino(dados, destino_override)
    log.info("Destino de indexacao: %s (%s)", destino, motivo_destino)
    if destino == "grafo":
        n = indexar_grafo(dados.get("conteudo", ""))
        return {"destino": destino, "motivo_destino": motivo_destino,
                "chunking": "grafo (LightRAG gerencia)", "motivo_chunking": "destino=grafo",
                "n_chunks": n}
    tecnica, motivo_chunking = avaliar_chunking(dados, chunking_override)
    log.info("Tecnica de chunking: %s (%s)", tecnica, motivo_chunking)
    docs = chunkar(dados.get("conteudo", ""), tecnica)
    n = indexar_opensearch(docs, meta)
    log.info("Indexacao concluida: %d chunk(s) no OpenSearch", n)
    return {"destino": destino, "motivo_destino": motivo_destino,
            "chunking": tecnica, "motivo_chunking": motivo_chunking, "n_chunks": n}
