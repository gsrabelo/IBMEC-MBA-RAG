"""
06_chat_langfuse.py - Chat RAG (Haystack) com observabilidade + rastreabilidade.

Chat de perguntas e respostas sobre o indice do RAPTOR (aula6_raptor por padrao).
Duas coisas acontecem a cada turno:

  1. OBSERVABILIDADE: a pergunta/resposta vira um TRACE no LangFuse
     (auto-instrumentacao via LangfuseConnector) - latencia, trechos, prompt, resposta.

  2. RASTREABILIDADE: para cada trecho recuperado, o chat mostra a ORIGEM:
       - se for um RESUMO de cluster -> de quais documentos ele foi gerado
       - se for um DOCUMENTO ORIGINAL -> qual o id e o cluster a que pertence

Pre-requisito: o indice precisa ter sido construido pelo RAPTOR (com os resumos):
    python 03_raptor.py --recriar
(Voce pode apontar para outro indice com --indice; sem resumos, mostra so o id.)

Precisa de OpenSearch, Ollama, Groq e (para tracing) LangFuse no .env.

Uso:
    python 06_chat_langfuse.py
    python 06_chat_langfuse.py --indice aula6_raptor --top-k 5
"""

import argparse

import _comum

_comum.carregar_env()  # antes de importar haystack (liga o tracing do LangFuse)

from haystack import Pipeline                                                 # noqa: E402
from haystack.components.builders import PromptBuilder                         # noqa: E402
from haystack.components.generators import OpenAIGenerator                     # noqa: E402
from haystack.utils import Secret                                             # noqa: E402
from haystack_integrations.components.retrievers.opensearch import (          # noqa: E402
    OpenSearchEmbeddingRetriever,
)

NOME_TRACE = "chat-rag-aula6"

TEMPLATE = """
Voce e um assistente juridico. Responda APENAS com base nos trechos abaixo, de
forma objetiva. Se nao constar, diga que nao consta.

Trechos:
{% for doc in documents %}
[{{ loop.index }}] {{ doc.content }}
{% endfor %}

Pergunta: {{ question }}
Resposta:
"""


def descrever(doc):
    """Descreve a origem do trecho: resumo de cluster, ou documento original."""
    if doc.meta.get("tipo") == "resumo_raptor":
        origem = doc.meta.get("documentos_origem", [])
        return f"RESUMO do cluster {doc.meta.get('cluster')} <- docs {origem}"
    cluster = doc.meta.get("cluster")
    base = f"ORIGINAL {doc.meta.get('id_original')}"
    return base + (f" (cluster {cluster})" if cluster is not None else "")


def montar_pipeline(store, top_k, usar_langfuse):
    base_url, modelo = _comum.config_ollama()
    groq_key, groq_modelo, groq_base = _comum.config_groq()

    pipe = Pipeline()
    if usar_langfuse:
        from haystack_integrations.components.connectors.langfuse import LangfuseConnector

        pipe.add_component("tracer", LangfuseConnector(NOME_TRACE))
    pipe.add_component("embedder", _comum.text_embedder())
    pipe.add_component("retriever", OpenSearchEmbeddingRetriever(document_store=store, top_k=top_k))
    pipe.add_component("prompt", PromptBuilder(template=TEMPLATE, required_variables=["documents", "question"]))
    pipe.add_component("llm", OpenAIGenerator(
        api_key=Secret.from_token(groq_key), model=groq_modelo, api_base_url=groq_base,
        generation_kwargs={"temperature": 0.2, "max_tokens": 500}))
    pipe.connect("embedder.embedding", "retriever.query_embedding")
    pipe.connect("retriever.documents", "prompt.documents")
    pipe.connect("prompt.prompt", "llm.prompt")
    return pipe


def main():
    parser = argparse.ArgumentParser(description="Chat RAG com LangFuse + rastreabilidade (Aula 6).")
    parser.add_argument("--indice", default="aula6_raptor", help="indice no OpenSearch")
    parser.add_argument("--top-k", type=int, default=5, help="quantos trechos usar")
    args = parser.parse_args()

    usar_langfuse = _comum.langfuse_configurado()
    print("=" * 60)
    print("  CHAT RAG + LANGFUSE (com rastreabilidade) - Aula 6")
    print("=" * 60)
    print(f"Indice: {args.indice} | LangFuse: {'ligado' if usar_langfuse else 'DESLIGADO (sem chaves no .env)'}")
    print(f"Nome do trace (para filtrar avaliadores): '{NOME_TRACE}'")

    store = _comum.abrir_store(args.indice)
    try:
        total = store.count_documents()
    except Exception:
        total = 0
    if total == 0:
        print(f"\n[ATENCAO] O indice '{args.indice}' esta vazio.")
        print("  Construa o indice do RAPTOR antes: python 03_raptor.py --recriar")
        return
    print(f"Documentos no indice: {total}")

    pipe = montar_pipeline(store, args.top_k, usar_langfuse)

    print("Digite sua pergunta. Para sair: 'sair' (ou Ctrl+C).\n")
    while True:
        try:
            pergunta = input("Pergunta> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nAte logo!")
            break
        if pergunta.lower() in {"sair", "exit", "quit"} or not pergunta:
            print("Ate logo!")
            break
        try:
            resultado = pipe.run(
                {"embedder": {"text": pergunta}, "prompt": {"question": pergunta}},
                include_outputs_from={"retriever", "tracer"})
            docs_r = resultado["retriever"]["documents"]
            resposta = resultado["llm"]["replies"][0]
            trace_url = resultado.get("tracer", {}).get("trace_url", "")
        except Exception as e:
            print(f"  [erro: {e}]\n")
            continue
        print("\nResposta:")
        print(f"  {resposta}")
        print("  Fontes (origem de cada trecho):")
        for i, d in enumerate(docs_r, 1):
            print(f"    {i}. {descrever(d)}")
        if trace_url:
            print(f"  Trace no LangFuse: {trace_url}")
        print()


if __name__ == "__main__":
    main()
