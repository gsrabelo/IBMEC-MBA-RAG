"""
04_naive_rag.py - Naive RAG sobre os PDFs indexados (Haystack + OpenSearch + Groq).

Fluxo de pergunta-e-resposta da Aula 2:
  1. Transforma a pergunta em vetor (Ollama).
  2. Busca os chunks mais parecidos no OpenSearch (busca vetorial kNN).
  3. Monta um prompt com esses chunks.
  4. Pede a resposta para um LLM da Groq, que responde SO com base nos documentos.

"Naive RAG" = o pipeline RAG mais simples (buscar + responder), sem reranking nem
reescrita de query. E a base sobre a qual as proximas aulas vao melhorar.

Antes de rodar, indexe os PDFs com:  python 03_indexar_chunks_opensearch.py

Precisa de OpenSearch, Ollama e da chave da Groq (veja 00_check_ambiente.py).

Uso:
    python 04_naive_rag.py --pergunta "Quais os requisitos da prisao preventiva?"
    python 04_naive_rag.py --pergunta "O que diz o manual sobre oitiva?" --top-k 5
"""

import argparse

from haystack import Pipeline
from haystack.components.builders import PromptBuilder
from haystack.components.generators import OpenAIGenerator
from haystack.utils import Secret
from haystack_integrations.components.embedders.ollama import OllamaTextEmbedder
from haystack_integrations.components.retrievers.opensearch import (
    OpenSearchEmbeddingRetriever,
)
from haystack_integrations.document_stores.opensearch import OpenSearchDocumentStore

import _comum

# Regra de ouro do RAG: o LLM so pode usar os documentos recuperados.
TEMPLATE = """
Voce e um assistente juridico especializado em direito e seguranca publica.
Responda APENAS com base nos trechos de documentos abaixo.
Se a informacao nao estiver nos trechos, diga claramente: "Essa informacao nao consta nos documentos disponiveis."
Cite a fonte de cada informacao (nome do arquivo). NAO invente leis nem fatos.

Trechos recuperados:
{% for doc in documents %}
[{{ loop.index }}] (fonte: {{ doc.meta.fonte }})
{{ doc.content }}
{% endfor %}

Pergunta: {{ question }}

Resposta fundamentada (com as fontes):
"""


def montar_pipeline(indice, top_k):
    """Cria e conecta os 4 componentes do pipeline Naive RAG."""
    _comum.carregar_env()
    base_url, modelo_embed = _comum.config_ollama()
    os_cfg = _comum.config_opensearch()
    auth = (os_cfg["usuario"], os_cfg["senha"]) if os_cfg["usuario"] else None
    groq_key, groq_modelo, groq_base = _comum.config_groq()
    dim = _comum.dimensao_do_modelo(modelo_embed)

    store = OpenSearchDocumentStore(
        hosts=os_cfg["url"], index=indice, embedding_dim=dim,
        http_auth=auth, use_ssl=False, verify_certs=False,
    )

    pipe = Pipeline()
    pipe.add_component("embedder", OllamaTextEmbedder(model=modelo_embed, url=base_url))
    pipe.add_component("retriever", OpenSearchEmbeddingRetriever(document_store=store, top_k=top_k))
    pipe.add_component("prompt", PromptBuilder(template=TEMPLATE, required_variables=["documents", "question"]))
    pipe.add_component("llm", OpenAIGenerator(
        api_key=Secret.from_token(groq_key),
        model=groq_modelo,
        api_base_url=groq_base,
        generation_kwargs={"temperature": 0.2, "max_tokens": 500},
    ))

    pipe.connect("embedder.embedding", "retriever.query_embedding")
    pipe.connect("retriever.documents", "prompt.documents")
    pipe.connect("prompt.prompt", "llm.prompt")
    return pipe


def main():
    parser = argparse.ArgumentParser(description="Naive RAG (Haystack + OpenSearch + Groq).")
    parser.add_argument("--pergunta", required=True, help="a pergunta a responder")
    parser.add_argument("--indice", default="aula2_juridico", help="indice no OpenSearch")
    parser.add_argument("--top-k", type=int, default=4, help="quantos chunks buscar")
    args = parser.parse_args()

    print("=" * 60)
    print("  NAIVE RAG (Haystack -> OpenSearch -> Groq) - Aula 2")
    print("=" * 60)
    print(f"Pergunta: {args.pergunta}")

    pipe = montar_pipeline(args.indice, args.top_k)
    resultado = pipe.run(
        {"embedder": {"text": args.pergunta}, "prompt": {"question": args.pergunta}},
        include_outputs_from={"retriever"},
    )

    print("\nTRECHOS RECUPERADOS (fontes):")
    print("-" * 50)
    for i, doc in enumerate(resultado["retriever"]["documents"], start=1):
        fonte = doc.meta.get("fonte", "?")
        print(f"  [{i}] (score {doc.score:.3f}) {fonte}")

    print("\nRESPOSTA DO LLM:")
    print("-" * 50)
    print(resultado["llm"]["replies"][0])


if __name__ == "__main__":
    main()
