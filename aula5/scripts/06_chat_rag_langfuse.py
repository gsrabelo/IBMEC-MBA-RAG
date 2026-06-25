"""
06_chat_rag_langfuse.py - Chat RAG instrumentado no LangFuse (para LLM-as-a-Judge).

Este e um chat de perguntas e respostas (RAG hibrido: BM25 + densa + RRF -> Groq).
A novidade: CADA pergunta/resposta vira um TRACE no LangFuse (auto-instrumentacao
via LangfuseConnector). Com os traces no LangFuse, voce configura na interface o
recurso de LLM-as-a-Judge (avaliadores Ragas: Hallucination, Context-Relevance,
Answer-Relevance) para o proprio LangFuse PONTUAR as respostas automaticamente.

Ou seja: aqui o script so PRODUZ os traces; a avaliacao acontece dentro do LangFuse.
Veja o passo a passo em GUIA_LANGFUSE_LLM_AS_JUDGE.md.

Precisa de OpenSearch (indice indexado), Ollama, Groq e LangFuse configurado no .env.

Uso:
    python 06_chat_rag_langfuse.py
    python 06_chat_rag_langfuse.py --indice aula4_hibrido --top-k 5
"""

import argparse

import _comum

_comum.carregar_env()  # antes de importar haystack (liga o tracing do LangFuse)

from haystack import Pipeline                                                 # noqa: E402
from haystack.components.builders import PromptBuilder                         # noqa: E402
from haystack.components.generators import OpenAIGenerator                     # noqa: E402
from haystack.components.joiners import DocumentJoiner                         # noqa: E402
from haystack.utils import Secret                                             # noqa: E402
from haystack_integrations.components.embedders.ollama import OllamaTextEmbedder  # noqa: E402
from haystack_integrations.components.retrievers.opensearch import (          # noqa: E402
    OpenSearchBM25Retriever,
    OpenSearchEmbeddingRetriever,
)
from haystack_integrations.document_stores.opensearch import OpenSearchDocumentStore  # noqa: E402

# Nome do trace: use este nome para FILTRAR o avaliador no LangFuse.
NOME_TRACE = "chat-rag-aula5"

TEMPLATE = """
Voce e um assistente juridico especializado em controle externo (TCU).
Responda APENAS com base nos trechos abaixo, de forma objetiva (1-3 frases).
Se a informacao nao estiver neles, diga que nao consta. Cite o titulo da fonte.

Trechos:
{% for doc in documents %}
[{{ loop.index }}] {{ doc.meta.titulo }}
{{ doc.content }}
{% endfor %}

Pergunta: {{ question }}
Resposta:
"""


def montar_pipeline(indice, top_k, usar_langfuse):
    base_url, modelo = _comum.config_ollama()
    os_cfg = _comum.config_opensearch()
    auth = (os_cfg["usuario"], os_cfg["senha"]) if os_cfg["usuario"] else None
    groq_key, groq_modelo, groq_base = _comum.config_groq()

    store = OpenSearchDocumentStore(
        hosts=os_cfg["url"], index=indice, embedding_dim=_comum.dimensao_do_modelo(modelo),
        http_auth=auth, use_ssl=False, verify_certs=False,
    )

    pipe = Pipeline()
    # Conector do LangFuse: instrumenta TODO o pipeline (input, retrieval, output).
    if usar_langfuse:
        from haystack_integrations.components.connectors.langfuse import LangfuseConnector

        pipe.add_component("tracer", LangfuseConnector(NOME_TRACE))

    pipe.add_component("embedder", OllamaTextEmbedder(model=modelo, url=base_url))
    pipe.add_component("bm25", OpenSearchBM25Retriever(document_store=store, top_k=top_k))
    pipe.add_component("denso", OpenSearchEmbeddingRetriever(document_store=store, top_k=top_k))
    pipe.add_component("juntar", DocumentJoiner(join_mode="reciprocal_rank_fusion", top_k=top_k))
    pipe.add_component("prompt", PromptBuilder(template=TEMPLATE, required_variables=["documents", "question"]))
    pipe.add_component("llm", OpenAIGenerator(
        api_key=Secret.from_token(groq_key), model=groq_modelo, api_base_url=groq_base,
        generation_kwargs={"temperature": 0.2, "max_tokens": 500}))

    pipe.connect("embedder.embedding", "denso.query_embedding")
    pipe.connect("bm25.documents", "juntar.documents")
    pipe.connect("denso.documents", "juntar.documents")
    pipe.connect("juntar.documents", "prompt.documents")
    pipe.connect("prompt.prompt", "llm.prompt")
    return pipe


def responder(pipe, pergunta):
    resultado = pipe.run(
        {"embedder": {"text": pergunta}, "bm25": {"query": pergunta},
         "prompt": {"question": pergunta}},
        include_outputs_from={"juntar", "tracer"},
    )
    fontes = [d.meta.get("titulo", "")[:60] for d in resultado["juntar"]["documents"]]
    resposta = resultado["llm"]["replies"][0]
    trace_url = ""
    if "tracer" in resultado:
        trace_url = resultado["tracer"].get("trace_url", "")
    return resposta, fontes, trace_url


def main():
    parser = argparse.ArgumentParser(description="Chat RAG instrumentado no LangFuse.")
    parser.add_argument("--indice", default="aula4_hibrido", help="indice no OpenSearch")
    parser.add_argument("--top-k", type=int, default=5, help="quantos trechos usar")
    args = parser.parse_args()

    usar_langfuse = _comum.langfuse_configurado()

    print("=" * 60)
    print("  CHAT RAG + LANGFUSE (LLM-as-a-Judge) - Aula 5")
    print("=" * 60)
    print(f"Indice: {args.indice} | LangFuse: {'ligado' if usar_langfuse else 'DESLIGADO (sem chaves no .env)'}")
    if not usar_langfuse:
        print("Aviso: sem LangFuse os traces nao serao enviados. Veja GUIA_LANGFUSE_WINDOWS.md (Aula 3).")
    print(f"Nome do trace (use para filtrar o avaliador): '{NOME_TRACE}'")
    print("Digite sua pergunta. Para sair: 'sair' (ou Ctrl+C).\n")

    pipe = montar_pipeline(args.indice, args.top_k, usar_langfuse)

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
            resposta, fontes, trace_url = responder(pipe, pergunta)
        except Exception as e:
            print(f"  [erro: {e}]\n")
            continue
        print("\nResposta:")
        print(f"  {resposta}")
        print(f"  Fontes: {fontes}")
        if trace_url:
            print(f"  Trace no LangFuse: {trace_url}")
        print()

    print("\nProximo passo: no LangFuse, configure um avaliador LLM-as-a-Judge "
          f"(Ragas) filtrando pelos traces '{NOME_TRACE}'. Veja GUIA_LANGFUSE_LLM_AS_JUDGE.md.")


if __name__ == "__main__":
    main()
