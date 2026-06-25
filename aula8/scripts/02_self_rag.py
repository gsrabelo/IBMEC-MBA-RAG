"""
02_self_rag.py - Self-RAG (training-free) como PIPELINE Haystack.

O Self-RAG "de verdade" exige um modelo com fine-tuning especifico. Como usamos a
Groq, fazemos a versao TRAINING-FREE: o LLM EMITE os tokens de controle por prompting.
Aqui TODO o fluxo e um pipeline Haystack, entao a auto-instrumentacao do LangFuse
captura cada etapa no mesmo trace ('self-rag-aula8').

Fluxo (linear / feed-forward):
  [Retrieve] (LLM yes/no)
     -> ConditionalRouter
         yes -> embedder -> retriever -> [ISREL] FiltrarRelevantes
         no  -> (sem busca)
     -> MontarContextoSelf (vazio se [Retrieve]=no)
     -> geracao da resposta (LLM)
     -> [ISSUP]/[ISUSE] (LLM critica a propria resposta)

Tokens: [Retrieve] yes/no | [ISREL] relevante/irrelevante por doc | [ISSUP]
fully/partially/no | [ISUSE] 1-5.

Precisa de OpenSearch (indice do TCU), Ollama e Groq.

Uso:
    python 02_self_rag.py --pergunta "o gestor pode ser multado pelo TCU?"
    python 02_self_rag.py --pergunta "o que e responsabilidade civil?"   # tende a [Retrieve=no]
    python 02_self_rag.py --pergunta "..." --sem-langfuse
"""

import argparse

import _comum

_comum.carregar_env()  # antes de importar haystack (liga o tracing do LangFuse)

from haystack import Pipeline                                                 # noqa: E402
from haystack.components.builders import PromptBuilder                         # noqa: E402
from haystack.components.generators import OpenAIGenerator                     # noqa: E402
from haystack.components.routers import ConditionalRouter                      # noqa: E402
from haystack.utils import Secret                                            # noqa: E402
from haystack_integrations.components.retrievers.opensearch import (          # noqa: E402
    OpenSearchEmbeddingRetriever,
)

from _componentes import FiltrarRelevantes, MontarContextoSelf, ParseRetrieve  # noqa: E402

RETRIEVE_TEMPLATE = (
    "Decida se responder a pergunta exige consultar documentos especificos "
    "(jurisprudencia, leis, acordaos) ou se e conhecimento geral.\n"
    "Responda APENAS JSON: {\"retrieve\": \"yes\"|\"no\", \"motivo\": \"<curto>\"}\n\n"
    "Pergunta: {{question}}"
)
RESPOSTA_TEMPLATE = """
Voce e um assistente juridico especializado em controle externo (TCU).
{% if documents %}Responda com base nos trechos abaixo, de forma objetiva. Se nao
constar, diga que nao consta.

Trechos:
{% for doc in documents %}
[{{ loop.index }}] {{ doc.content }}
{% endfor %}
{% else %}Nenhum documento foi recuperado ([Retrieve]=no). Responda com seu
conhecimento geral, de forma objetiva e cautelosa.
{% endif %}
Pergunta: {{ question }}
Resposta:
"""
CRITICA_TEMPLATE = (
    "Avalie a RESPOSTA gerada para a pergunta, em relacao aos TRECHOS.\n"
    "Responda APENAS JSON: {\"issup\": \"fully\"|\"partially\"|\"no\", "
    "\"isuse\": <1-5>}\n"
    "- issup: a resposta tem suporte factual nos trechos? (sem trechos -> avalie se "
    "a resposta se sustenta sozinha)\n- isuse: utilidade da resposta (1 a 5)\n\n"
    "Trechos:\n{% for doc in documents %}- {{ doc.content }}\n{% endfor %}\n"
    "Pergunta: {{ question }}\nResposta: {{ answer[0] }}"
)

ROUTES = [
    {"condition": "{{retrieve == 'yes'}}", "output": "{{question}}",
     "output_name": "q_busca", "output_type": str},
    {"condition": "{{retrieve == 'no'}}", "output": "{{question}}",
     "output_name": "q_direto", "output_type": str},
]


def _llm(temp=0.2, max_tokens=500):
    groq_key, groq_modelo, groq_base = _comum.config_groq()
    return OpenAIGenerator(api_key=Secret.from_token(groq_key), model=groq_modelo,
                           api_base_url=groq_base,
                           generation_kwargs=_comum.gen_kwargs(groq_modelo, temp, max_tokens))


def montar_pipeline(store, top_k, usar_langfuse, limite_rel=0.5):
    pipe = Pipeline(max_runs_per_component=10)
    if usar_langfuse:
        from haystack_integrations.components.connectors.langfuse import LangfuseConnector

        pipe.add_component("tracer", LangfuseConnector("self-rag-aula8"))

    # [Retrieve]
    pipe.add_component("retrieve_prompt", PromptBuilder(template=RETRIEVE_TEMPLATE,
                                                        required_variables=["question"]))
    pipe.add_component("retrieve_llm", _llm(temp=0.0, max_tokens=80))
    pipe.add_component("parse", ParseRetrieve())
    pipe.add_component("router", ConditionalRouter(routes=ROUTES))
    # ramo yes: busca + [ISREL]
    pipe.add_component("embedder", _comum.text_embedder())
    pipe.add_component("retriever", OpenSearchEmbeddingRetriever(document_store=store, top_k=top_k))
    pipe.add_component("filtrar", FiltrarRelevantes(limite=limite_rel))
    pipe.add_component("montar", MontarContextoSelf())
    # geracao
    pipe.add_component("gen_prompt", PromptBuilder(template=RESPOSTA_TEMPLATE,
                                                   required_variables=["documents", "question"]))
    pipe.add_component("gen_llm", _llm(temp=0.2, max_tokens=500))
    # [ISSUP]/[ISUSE]
    pipe.add_component("critica_prompt", PromptBuilder(template=CRITICA_TEMPLATE,
                                                       required_variables=["documents", "question", "answer"]))
    pipe.add_component("critica_llm", _llm(temp=0.0, max_tokens=80))

    pipe.connect("retrieve_prompt.prompt", "retrieve_llm.prompt")
    pipe.connect("retrieve_llm.replies", "parse.replies")
    pipe.connect("parse.retrieve", "router.retrieve")
    pipe.connect("parse.question", "router.question")
    pipe.connect("router.q_busca", "embedder.text")
    pipe.connect("embedder.embedding", "retriever.query_embedding")
    pipe.connect("retriever.documents", "filtrar.documents")
    pipe.connect("parse.question", "filtrar.question")
    pipe.connect("parse.question", "montar.question")
    pipe.connect("filtrar.documents", "montar.documents")
    pipe.connect("montar.documents", "gen_prompt.documents")
    pipe.connect("parse.question", "gen_prompt.question")
    pipe.connect("gen_prompt.prompt", "gen_llm.prompt")
    pipe.connect("montar.documents", "critica_prompt.documents")
    pipe.connect("parse.question", "critica_prompt.question")
    pipe.connect("gen_llm.replies", "critica_prompt.answer")
    pipe.connect("critica_prompt.prompt", "critica_llm.prompt")
    return pipe


def responder(pipe, pergunta, usar_langfuse):
    saidas = {"parse", "filtrar", "montar", "gen_llm", "critica_llm"}
    if usar_langfuse:
        saidas.add("tracer")
    return pipe.run(
        {"retrieve_prompt": {"question": pergunta}, "parse": {"question": pergunta}},
        include_outputs_from=saidas)


def main():
    parser = argparse.ArgumentParser(description="Self-RAG training-free em Haystack (Aula 8).")
    parser.add_argument("--pergunta", required=True)
    parser.add_argument("--indice", default=_comum.INDICE_TCU)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--sem-langfuse", action="store_true")
    args = parser.parse_args()

    usar_langfuse = _comum.langfuse_configurado() and not args.sem_langfuse

    print("=" * 60)
    print("  SELF-RAG (training-free) em Haystack - Aula 8")
    print("=" * 60)
    print(f"Pergunta: {args.pergunta}")
    print(f"LangFuse: {'ligado (trace self-rag-aula8)' if usar_langfuse else 'desligado'}\n")

    store = _comum.abrir_store(args.indice)
    try:
        if store.count_documents() == 0:
            print("[ATENCAO] Indice vazio. Rode antes: python 01_indexar_opensearch.py")
            return
    except Exception as e:
        print(f"[ATENCAO] nao consegui acessar o indice: {e}")
        return

    pipe = montar_pipeline(store, args.top_k, usar_langfuse)
    r = responder(pipe, args.pergunta, usar_langfuse)

    retrieve = r["parse"]["retrieve"]
    print(f"[Retrieve] = {retrieve}")
    if "filtrar" in r:
        print("[ISREL] relevancia por documento:")
        for a in r["filtrar"]["avaliacoes"]:
            print(f"   - {a['id']}: score={a['score']:.2f} -> "
                  f"{'relevant' if a['relevante'] else 'irrelevant'}")
    else:
        print("   (sem busca; respondendo por conhecimento geral)")

    docs = r["montar"]["documents"]
    resposta = r["gen_llm"]["replies"][0]
    print(f"\nResposta:\n{resposta}")

    critica = _comum.extrair_json(r["critica_llm"]["replies"][0])
    print(f"\n[ISSUP] = {critica.get('issup', '?')}")
    print(f"[ISUSE] = {critica.get('isuse', '?')}/5")
    print(f"Fontes usadas: {[d.meta.get('id_original') for d in docs]}")
    if usar_langfuse:
        url = r.get("tracer", {}).get("trace_url", "")
        if url:
            print(f"\nTrace LangFuse: {url}")


if __name__ == "__main__":
    main()
