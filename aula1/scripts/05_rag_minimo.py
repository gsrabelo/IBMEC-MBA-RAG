"""
05_rag_minimo.py - Pipeline RAG minimo com Haystack.

Junta tudo da Aula 1 em um fluxo de pergunta-e-resposta:
  1. Transforma a pergunta em vetor (Ollama).
  2. Busca os documentos mais parecidos no OpenSearch (busca vetorial kNN).
  3. Monta um prompt com esses documentos.
  4. Pede a resposta para um LLM da Groq, que responde SO com base nos documentos.

Antes de rodar este script, indexe o corpus com:  python 04_indexar_opensearch.py

Precisa de OpenSearch, Ollama e da chave da Groq (veja 00_check_ambiente.py).

Uso:
    python 05_rag_minimo.py --pergunta "Quais os requisitos da prisao preventiva?"
    python 05_rag_minimo.py --pergunta "O que e peculato?" --top-k 5
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

# Instrucoes para o LLM. A regra de ouro do RAG: so usar os documentos dados.
TEMPLATE = """
Voce e um assistente juridico especializado em direito penal brasileiro.
Responda APENAS com base nos documentos abaixo.
Se a informacao nao estiver nos documentos, diga claramente que nao consta.
Cite o titulo do documento que usou. NAO invente leis nem fatos.

Documentos:
{% for doc in documents %}
[{{ loop.index }}] {{ doc.meta.titulo }} (categoria: {{ doc.meta.categoria }})
{{ doc.content }}
{% endfor %}

Pergunta: {{ question }}

Resposta fundamentada:
"""


def montar_pipeline(indice, top_k):
    """Cria e conecta os 4 componentes do pipeline RAG."""
    _comum.carregar_env()
    base_url, modelo_embed = _comum.config_ollama()
    os_cfg = _comum.config_opensearch()
    auth = (os_cfg["usuario"], os_cfg["senha"]) if os_cfg["usuario"] else None
    groq_key, groq_modelo, groq_base = _comum.config_groq()
    dim = _comum.dimensao_do_modelo(modelo_embed)

    store = OpenSearchDocumentStore(
        hosts=os_cfg["url"],
        index=indice,
        embedding_dim=dim,
        http_auth=auth,
        use_ssl=False,
        verify_certs=False,
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

    # Liga as saidas de um componente nas entradas do proximo.
    pipe.connect("embedder.embedding", "retriever.query_embedding")
    pipe.connect("retriever.documents", "prompt.documents")
    pipe.connect("prompt.prompt", "llm.prompt")
    return pipe


def main():
    parser = argparse.ArgumentParser(description="Pipeline RAG minimo (Haystack + Groq).")
    parser.add_argument("--pergunta", required=True, help="a pergunta a responder")
    parser.add_argument("--indice", default="aula1_juridico", help="indice no OpenSearch")
    parser.add_argument("--top-k", type=int, default=3, help="quantos documentos buscar")
    args = parser.parse_args()

    print("=" * 60)
    print("  RAG MINIMO (Haystack -> OpenSearch -> Groq) - Aula 1")
    print("=" * 60)
    print(f"Pergunta: {args.pergunta}")

    pipe = montar_pipeline(args.indice, args.top_k)
    resultado = pipe.run(
        {"embedder": {"text": args.pergunta}, "prompt": {"question": args.pergunta}},
        include_outputs_from={"retriever"},
    )

    # Mostra os documentos que o sistema encontrou (as "fontes" da resposta).
    print("\nDOCUMENTOS RECUPERADOS (fontes):")
    print("-" * 50)
    for i, doc in enumerate(resultado["retriever"]["documents"], start=1):
        print(f"  [{i}] (score {doc.score:.3f}) {doc.meta['titulo']}")

    # Mostra a resposta final do LLM.
    print("\nRESPOSTA DO LLM:")
    print("-" * 50)
    print(resultado["llm"]["replies"][0])


if __name__ == "__main__":
    main()
