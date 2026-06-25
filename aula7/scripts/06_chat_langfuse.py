"""
06_chat_langfuse.py - Chat que testa as TECNICAS de Query Enhancement + LangFuse.

Cada tecnica e um PIPELINE HAYSTACK COMPLETO - inclusive a geracao de variacoes /
step-back, feita por componentes OpenAIGenerator. Assim a auto-instrumentacao do
LangFuse captura TODOS os passos no mesmo trace: a chamada de LLM que reescreve a
pergunta, a busca/fusao, o prompt final e a resposta. Cada tecnica tem um trace
com nome proprio ('chat-aula7-<tecnica>') para comparar lado a lado.

Tecnicas:
  - baseline    : 1 busca (sem enhancement)
  - multi_query : LLM gera variacoes -> busca cada -> dedup
  - step_back   : LLM gera pergunta geral -> busca (especifica + geral) -> dedup
  - rag_fusion  : LLM gera variacoes -> busca cada -> RRF

Precisa de OpenSearch (indice do TCU), Ollama, Groq e (para tracing) LangFuse.

Uso:
    python 06_chat_langfuse.py                 # roda TODAS as tecnicas por pergunta
    python 06_chat_langfuse.py --tecnica rag_fusion
    python 06_chat_langfuse.py --n 4 --top-k 5
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

from _componentes import BuscarMultiplas, MontarConsultas                      # noqa: E402

TECNICAS = ["baseline", "multi_query", "step_back", "rag_fusion"]

TEMPLATE_RESPOSTA = """
Voce e um assistente juridico especializado em controle externo (TCU).
Responda APENAS com base nos trechos abaixo, de forma objetiva. Se nao constar,
diga que nao consta.

Trechos:
{% for doc in documents %}
[{{ loop.index }}] {{ doc.content }}
{% endfor %}

Pergunta: {{ question }}
Resposta:
"""
SB_TEMPLATE = ("Dada a pergunta especifica abaixo, formule UMA pergunta mais GERAL sobre "
               "o conceito juridico por tras dela. Responda apenas com a pergunta geral.\n\n"
               "Pergunta especifica: {{question}}")


def _llm():
    groq_key, groq_modelo, groq_base = _comum.config_groq()
    return OpenAIGenerator(api_key=Secret.from_token(groq_key), model=groq_modelo,
                           api_base_url=groq_base,
                           generation_kwargs={"temperature": 0.2, "max_tokens": 500})


def _llm_reescrita(temp):
    groq_key, groq_modelo, groq_base = _comum.config_groq()
    return OpenAIGenerator(api_key=Secret.from_token(groq_key), model=groq_modelo,
                           api_base_url=groq_base,
                           generation_kwargs={"temperature": temp, "max_tokens": 300})


def montar_pipeline(tecnica, store, top_k, n, usar_langfuse):
    """Monta um pipeline Haystack COMPLETO para a tecnica (tudo traçado no LangFuse)."""
    base_url, modelo = _comum.config_ollama()
    pipe = Pipeline()
    if usar_langfuse:
        from haystack_integrations.components.connectors.langfuse import LangfuseConnector

        pipe.add_component("tracer", LangfuseConnector(f"chat-aula7-{tecnica}"))
    pipe.add_component("prompt", PromptBuilder(template=TEMPLATE_RESPOSTA, required_variables=["documents", "question"]))
    pipe.add_component("llm", _llm())
    pipe.connect("prompt.prompt", "llm.prompt")

    if tecnica == "baseline":
        pipe.add_component("embedder", _comum.text_embedder())
        pipe.add_component("retriever", OpenSearchEmbeddingRetriever(document_store=store, top_k=top_k))
        pipe.connect("embedder.embedding", "retriever.query_embedding")
        pipe.connect("retriever.documents", "prompt.documents")
        return pipe

    # tecnicas com reescrita de query (LLM como componente -> entra no trace)
    if tecnica == "step_back":
        pipe.add_component("sb_prompt", PromptBuilder(template=SB_TEMPLATE, required_variables=["question"]))
        pipe.add_component("sb_llm", _llm_reescrita(0.3))
        pipe.add_component("montar", MontarConsultas(modo="stepback"))
        pipe.connect("sb_prompt.prompt", "sb_llm.prompt")
        pipe.connect("sb_llm.replies", "montar.textos")
    else:  # multi_query / rag_fusion
        var_template = (f"Gere {n} variacoes da pergunta juridica abaixo, com vocabulario "
                        "diferente (sinonimos, termos tecnicos). Uma por linha, sem numeracao."
                        "\n\nPergunta: " + "{{question}}")
        pipe.add_component("var_prompt", PromptBuilder(template=var_template, required_variables=["question"]))
        pipe.add_component("var_llm", _llm_reescrita(0.7))
        pipe.add_component("montar", MontarConsultas(modo="variacoes", n=n))
        pipe.connect("var_prompt.prompt", "var_llm.prompt")
        pipe.connect("var_llm.replies", "montar.textos")

    modo = "rrf" if tecnica == "rag_fusion" else "dedup"
    pipe.add_component("buscar", BuscarMultiplas(document_store=store, top_k=top_k, modo=modo))
    pipe.connect("montar.queries", "buscar.queries")
    pipe.connect("buscar.documents", "prompt.documents")
    return pipe


def entradas(tecnica, q):
    if tecnica == "baseline":
        return {"embedder": {"text": q}, "prompt": {"question": q}}
    rep = "sb_prompt" if tecnica == "step_back" else "var_prompt"
    return {rep: {"question": q}, "montar": {"question": q}, "prompt": {"question": q}}


def main():
    parser = argparse.ArgumentParser(description="Chat testando tecnicas + LangFuse (Aula 7).")
    parser.add_argument("--indice", default=_comum.INDICE_TCU)
    parser.add_argument("--tecnica", default="todas", choices=TECNICAS + ["todas"])
    parser.add_argument("--n", type=int, default=4, help="variacoes (multi_query/rag_fusion)")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    usar_langfuse = _comum.langfuse_configurado()
    selecionadas = TECNICAS if args.tecnica == "todas" else [args.tecnica]

    print("=" * 60)
    print("  CHAT - TESTE DE TECNICAS + LANGFUSE - Aula 7")
    print("=" * 60)
    print(f"Indice: {args.indice} | Tecnicas: {selecionadas}")
    print(f"LangFuse: {'ligado' if usar_langfuse else 'DESLIGADO (sem chaves no .env)'}")
    print("Pipelines completos: a reescrita de query (LLM) tambem entra no trace.")

    store = _comum.abrir_store(args.indice)
    try:
        if store.count_documents() == 0:
            print("\n[ATENCAO] Indice vazio. Rode antes: python 01_indexar_opensearch.py")
            return
    except Exception as e:
        print(f"[ATENCAO] nao consegui acessar o indice: {e}")
        return

    pipelines = {t: montar_pipeline(t, store, args.top_k, args.n, usar_langfuse) for t in selecionadas}

    print("\nDigite sua pergunta. Para sair: 'sair' (ou Ctrl+C).\n")
    while True:
        try:
            pergunta = input("Pergunta> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nAte logo!")
            break
        if pergunta.lower() in {"sair", "exit", "quit"} or not pergunta:
            print("Ate logo!")
            break
        for tecnica in selecionadas:
            chave_docs = "retriever" if tecnica == "baseline" else "buscar"
            try:
                resultado = pipelines[tecnica].run(
                    entradas(tecnica, pergunta),
                    include_outputs_from={"tracer", chave_docs})
                docs = resultado.get(chave_docs, {}).get("documents", [])
                resposta = resultado["llm"]["replies"][0]
                trace_url = resultado.get("tracer", {}).get("trace_url", "")
            except Exception as e:
                print(f"  [{tecnica}] erro: {e}")
                continue
            print(f"\n### {tecnica}")
            print(f"  Fontes: {[d.meta.get('id_original') for d in docs]}")
            print(f"  Resposta: {resposta[:400]}")
            if trace_url:
                print(f"  Trace: {trace_url}")
        print()


if __name__ == "__main__":
    main()
