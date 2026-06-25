"""
_componentes.py - Componentes Haystack customizados da Aula 7.

Servem para colocar TODA a tecnica (incluindo a geracao de variacoes/step-back)
DENTRO de um unico pipeline Haystack. Assim, com a auto-instrumentacao do LangFuse,
cada passo vira um span do trace - inclusive as chamadas de LLM que reescrevem a
pergunta (que antes ficavam "fora" do trace).

  - MontarConsultas : transforma a saida do LLM (texto) na lista de consultas
                      ([pergunta original] + variacoes) ou ([pergunta, pergunta geral]).
  - BuscarMultiplas : recebe a lista de consultas, busca cada uma no OpenSearch e
                      funde os resultados (dedup por score ou RRF).
"""

from typing import List

from haystack import Document, component

import _comum


@component
class MontarConsultas:
    """Monta a lista de consultas a partir da pergunta original + saida do LLM."""

    def __init__(self, modo="variacoes", n=4):
        self.modo = modo      # "variacoes" (multi-query/rag-fusion) ou "stepback"
        self.n = n

    @component.output_types(queries=List[str])
    def run(self, question: str, textos: List[str]):
        texto = textos[0] if textos else ""
        if self.modo == "stepback":
            queries = [question] + ([texto.strip()] if texto.strip() else [])
        else:
            variacoes = [v.strip(" -.") for v in texto.splitlines() if v.strip()][: self.n]
            queries = [question] + variacoes
        return {"queries": queries}


@component
class BuscarMultiplas:
    """Busca cada consulta no OpenSearch (Ollama) e funde (dedup por score ou RRF)."""

    def __init__(self, document_store, top_k=5, modo="dedup"):
        from haystack_integrations.components.embedders.ollama import OllamaTextEmbedder
        from haystack_integrations.components.retrievers.opensearch import (
            OpenSearchEmbeddingRetriever,
        )

        base_url, modelo = _comum.config_ollama()
        self.embedder = OllamaTextEmbedder(model=modelo, url=base_url)
        self.retriever = OpenSearchEmbeddingRetriever(document_store=document_store, top_k=top_k)
        self.top_k = top_k
        self.modo = modo      # "dedup" (multi-query/step-back) ou "rrf" (rag-fusion)

    def warm_up(self):
        if hasattr(self.embedder, "warm_up"):
            self.embedder.warm_up()

    @component.output_types(documents=List[Document])
    def run(self, queries: List[str]):
        listas = []
        for q in queries:
            emb = self.embedder.run(text=q)["embedding"]
            listas.append(self.retriever.run(query_embedding=emb)["documents"])
        if self.modo == "rrf":
            docs = _comum.fundir_rrf(listas, self.top_k)
        else:
            docs = _comum.dedup_por_id(listas, self.top_k)
        return {"documents": docs}
