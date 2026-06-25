"""
modelos.py - Schemas Pydantic da API (entrada/saida dos endpoints).
"""

from typing import Optional

from pydantic import BaseModel, Field


class RelatorioIngestao(BaseModel):
    """O que a API devolve apos ingerir um documento (mostra a DECISAO tomada)."""
    arquivo: str
    complexidade: str = Field(..., description="planilha | texto_simples | complexo | imagem")
    tecnica_extracao: str = Field(..., description="ferramenta escolhida pelo agente")
    motivo_extracao: str = ""
    estrutura: dict = Field(default_factory=dict, description="sinais do documento")
    destino: str = Field(..., description="opensearch | grafo (LightRAG)")
    motivo_destino: str = ""
    chunking: str = Field("", description="tecnica de chunking escolhida (so no OpenSearch)")
    motivo_chunking: str = ""
    n_chunks: int = 0
    n_caracteres: int = 0


class ConsultaRequest(BaseModel):
    pergunta: str
    destino: str = Field("auto", description="auto | opensearch | grafo")
    top_k: int = 5


class ConsultaResponse(BaseModel):
    pergunta: str
    resposta: str
    destino_usado: str
    fontes: list = Field(default_factory=list)


class IngestaoResponse(BaseModel):
    ok: bool
    relatorio: Optional[RelatorioIngestao] = None
    erro: Optional[str] = None
