"""
main.py - API FastAPI do Projeto Final (Aula 12).

Endpoints:
  POST /ingestao  - envia um documento (PDF/DOCX/XLSX/imagem/TXT). O AGENTE decide a
                    tecnica de extracao; a heuristica decide a estrategia/destino de
                    indexacao (OpenSearch ou LightRAG). Retorna o RELATORIO da decisao.
  POST /consulta  - pergunta -> resposta RAG (roteada para o storage certo) + fontes.
  GET  /health    - status dos componentes.
  GET  /metrics   - contadores simples (ingestoes, consultas, erros).

Autenticacao opcional por API key (header X-API-Key) se API_KEYS estiver no .env.

Rodar:  uvicorn app.main:app --reload    (de dentro de projeto_final/)
"""

import time

from fastapi import FastAPI, File, Header, HTTPException, UploadFile

from . import config, consulta, extracao, indexacao
from .log import configurar_logging, obter_logger
from .modelos import (ConsultaRequest, ConsultaResponse, IngestaoResponse,
                      RelatorioIngestao)

configurar_logging()           # le LOG_LEVEL do .env (DEBUG p/ verbosidade maxima)
log = obter_logger(__name__)

app = FastAPI(title="RAG Juridico - Projeto Final (Aula 12)",
              description="Ingestao inteligente (agente decide extracao) + RAG (OpenSearch/LightRAG)")

METRICAS = {"ingestoes": 0, "consultas": 0, "erros": 0, "inicio": time.time()}


def _checar_api_key(x_api_key):
    chaves = config.api_keys()
    if chaves and x_api_key not in chaves:
        raise HTTPException(status_code=401, detail="API key invalida (header X-API-Key).")


@app.post("/ingestao", response_model=IngestaoResponse)
async def ingestao(arquivo: UploadFile = File(...), estrategia: str = "auto",
                   chunking: str = "auto", x_api_key: str = Header(default="")):
    """Recebe um documento, decide como extrair e indexar, e devolve o relatorio.

    estrategia: auto | opensearch | grafo   (destino da indexacao)
    chunking:   auto | fixo | recursivo | sentenca_janela | semantico | hierarquico
    """
    _checar_api_key(x_api_key)
    destino = config.PASTA_UPLOADS / arquivo.filename
    t0 = time.time()
    log.info("== /ingestao recebido: arquivo=%s (estrategia=%s, chunking=%s) ==",
             arquivo.filename, estrategia, chunking)
    try:
        destino.write_bytes(await arquivo.read())
        # 1) AGENTE decide a tecnica e extrai
        sinais, tecnica, complexidade, motivo, dados = extracao.decidir_e_extrair(str(destino))
        # 2) HEURISTICA decide destino + (no OpenSearch) a melhor tecnica de chunking, e indexa
        estr = indexacao.indexar(dados, meta={"arquivo": arquivo.filename},
                                 destino_override=estrategia, chunking_override=chunking)
        extracao.limpar_cache(str(destino))
        METRICAS["ingestoes"] += 1
        log.info("== /ingestao OK: arquivo=%s, destino=%s, chunking=%s, chunks=%d (%.1fs) ==",
                 arquivo.filename, estr["destino"], estr["chunking"], estr["n_chunks"],
                 time.time() - t0)
        relatorio = RelatorioIngestao(
            arquivo=arquivo.filename, complexidade=complexidade, tecnica_extracao=tecnica,
            motivo_extracao=motivo, estrutura=sinais,
            destino=estr["destino"], motivo_destino=estr["motivo_destino"],
            chunking=estr["chunking"], motivo_chunking=estr["motivo_chunking"],
            n_chunks=estr["n_chunks"], n_caracteres=len(dados.get("conteudo", "")))
        return IngestaoResponse(ok=True, relatorio=relatorio)
    except Exception as e:
        METRICAS["erros"] += 1
        log.exception("== /ingestao FALHOU: arquivo=%s ==", arquivo.filename)
        return IngestaoResponse(ok=False, erro=str(e))


@app.post("/consulta", response_model=ConsultaResponse)
def consulta_endpoint(req: ConsultaRequest, x_api_key: str = Header(default="")):
    """Consulta RAG: roteia para OpenSearch ou LightRAG e gera a resposta."""
    _checar_api_key(x_api_key)
    log.info("== /consulta recebida: destino=%s, top_k=%d ==", req.destino, req.top_k)
    try:
        resposta, fontes, destino = consulta.consultar(req.pergunta, req.destino, req.top_k)
        METRICAS["consultas"] += 1
        log.info("== /consulta OK: destino_usado=%s, %d fonte(s) ==", destino, len(fontes))
        return ConsultaResponse(pergunta=req.pergunta, resposta=resposta,
                                destino_usado=destino, fontes=fontes)
    except Exception as e:
        METRICAS["erros"] += 1
        log.exception("== /consulta FALHOU ==")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    """Status rapido dos componentes (sem derrubar a API se algo falhar)."""
    estado = {"api": "ok"}
    try:
        estado["opensearch"] = f"ok ({indexacao._store_opensearch().count_documents()} docs)"
    except Exception as e:
        estado["opensearch"] = f"falhou: {e}"
    api_key, modelo, _ = config.config_groq()
    estado["groq"] = "ok (chave presente)" if api_key else "sem GROQ_API_KEY"
    estado["embedding"] = config.config_ollama()[1]
    estado["langfuse"] = "on" if config.langfuse_configurado() else "off"
    return estado


@app.get("/metrics")
def metrics():
    m = dict(METRICAS)
    m["uptime_s"] = round(time.time() - METRICAS["inicio"], 1)
    return m
