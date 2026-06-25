"""
_componentes.py - Componentes Haystack customizados da Aula 8.

Usados para montar TODO o CRAG e o Self-RAG dentro de pipelines Haystack, de modo
que a auto-instrumentacao do LangFuse capture cada etapa no mesmo trace.

CRAG:
  - AvaliarRota    : LLM-as-Judge nos documentos locais -> score medio -> rota.
  - BuscaWeb       : web search via TavilyWebSearch OFICIAL (fallback offline).
  - MontarContexto : junta o contexto final conforme a rota (local/fusao/web).

Self-RAG (training-free, pipeline linear):
  - ParseRetrieve     : le a saida do LLM e decide [Retrieve] yes/no.
  - FiltrarRelevantes : [ISREL] - mantem so os documentos relevantes (LLM-as-Judge).
  - MontarContextoSelf: junta o contexto (vazio se [Retrieve]=no).
"""

from typing import List, Optional

from haystack import Document, component

import _comum


# ---------------------------------------------------------------------------
# CRAG
# ---------------------------------------------------------------------------
@component
class AvaliarRota:
    """Avalia relevancia (LLM-as-Judge) e decide a rota CRAG (local/fusao/web)."""

    @component.output_types(documents=List[Document], question=str, rota=str, score=float)
    def run(self, documents: List[Document], question: str):
        cliente, modelo = _comum.groq_client()
        media, _ = _comum.avaliar_documentos(cliente, modelo, question, documents)
        return {"documents": documents, "question": question,
                "rota": _comum.decidir_rota(media), "score": media}


@component
class BuscaWeb:
    """Web search via TavilyWebSearch OFICIAL do Haystack; cai p/ stub offline.

    So roda quando o ConditionalRouter envia a query (rota fusao/web).
    """

    def __init__(self, max_results=3):
        self.max_results = max_results
        self._tavily = _comum.criar_tavily(top_k=max_results)  # None se sem chave/pacote

    @staticmethod
    def _retag(d: Document) -> Document:
        url = d.meta.get("url") or d.meta.get("link") or "web"
        d.meta["tipo"] = "web"
        d.meta["id_original"] = url
        return d

    @component.output_types(web_docs=List[Document])
    def run(self, query: str):
        if self._tavily is not None:
            try:
                docs = self._tavily.run(query=query)["documents"]
                return {"web_docs": [self._retag(d) for d in docs]}
            except Exception as e:
                return {"web_docs": [Document(content=f"[WEB SEARCH FALHOU] {e}",
                                              meta={"tipo": "web", "id_original": "web"})]}
        return {"web_docs": [Document(
            content=("[WEB SEARCH OFFLINE] Sem TAVILY_API_KEY no .env (ou pacote "
                     "tavily-haystack ausente) - o fallback de web search nao foi "
                     "executado. Configure a chave para buscar na web."),
            meta={"tipo": "web", "id_original": "web"})]}


@component
class MontarContexto:
    """Monta o contexto final do CRAG de acordo com a rota."""

    @component.output_types(documents=List[Document])
    def run(self, documents_local: List[Document], rota: str,
            web_docs: Optional[List[Document]] = None):
        web_docs = web_docs or []
        if rota == "local":
            docs = documents_local
        elif rota == "web":
            docs = web_docs
        else:  # fusao
            docs = list(documents_local) + list(web_docs)
        return {"documents": docs}


# ---------------------------------------------------------------------------
# Self-RAG (training-free)
# ---------------------------------------------------------------------------
@component
class ParseRetrieve:
    """Le a saida do LLM do token [Retrieve] e devolve a decisao yes/no."""

    @component.output_types(retrieve=str, question=str)
    def run(self, replies: List[str], question: str):
        dados = _comum.extrair_json(replies[0] if replies else "")
        decisao = str(dados.get("retrieve", "yes")).lower()
        if decisao not in ("yes", "no"):
            decisao = "yes"
        return {"retrieve": decisao, "question": question}


@component
class FiltrarRelevantes:
    """[ISREL] - mantem so os documentos avaliados como relevantes (LLM-as-Judge)."""

    def __init__(self, limite=0.5):
        self.limite = limite

    @component.output_types(documents=List[Document], avaliacoes=List[dict])
    def run(self, documents: List[Document], question: str):
        cliente, modelo = _comum.groq_client()
        relevantes, avaliacoes = [], []
        for d in documents:
            score, motivo = _comum.avaliar_documento(cliente, modelo, question, d.content)
            rel = score >= self.limite
            avaliacoes.append({"id": d.meta.get("id_original"), "score": score, "relevante": rel})
            if rel:
                relevantes.append(d)
        return {"documents": relevantes, "avaliacoes": avaliacoes}


@component
class MontarContextoSelf:
    """Junta o contexto do Self-RAG; fica vazio quando [Retrieve]=no (sem busca)."""

    @component.output_types(documents=List[Document])
    def run(self, question: str, documents: Optional[List[Document]] = None):
        return {"documents": documents or []}
