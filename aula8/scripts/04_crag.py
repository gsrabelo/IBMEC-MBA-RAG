"""
04_crag.py - CRAG completo (Corrective RAG) com Haystack ConditionalRouter.

Fluxo (tudo dentro de UM pipeline Haystack):

  [embedder -> retriever]  recupera documentos locais (OpenSearch/Ollama)
        |
  [avaliar]  LLM-as-Judge da um score 0-1 -> decide a rota:
        |       score >= 0.7 -> 'local' | 0.3-0.7 -> 'fusao' | < 0.3 -> 'web'
  [router] (ConditionalRouter)
        |--- rota 'local' .................... usa so os documentos locais
        '--- rota 'fusao'/'web' -> [busca_web] (Tavily com fallback offline)
        |
  [montar]  monta o contexto final conforme a rota (local / local+web / web)
        |
  [prompt -> llm]  gera a resposta

Sem LangGraph: o roteamento condicional e feito pelo ConditionalRouter do Haystack.

Uso:
    python 04_crag.py --pergunta "quando as contas sao julgadas irregulares?"
    python 04_crag.py --pergunta "decisoes do STF em 2024 sobre interceptacao" --top-k 4
    python 04_crag.py --pergunta "..." --sem-langfuse
"""

import argparse
from typing import List

import _comum

_comum.carregar_env()  # antes de importar haystack (liga o tracing do LangFuse)

from haystack import Document, Pipeline                                        # noqa: E402
from haystack.components.builders import PromptBuilder                          # noqa: E402
from haystack.components.generators import OpenAIGenerator                      # noqa: E402
from haystack.components.routers import ConditionalRouter                       # noqa: E402
from haystack.utils import Secret                                             # noqa: E402
from haystack_integrations.components.retrievers.opensearch import (           # noqa: E402
    OpenSearchEmbeddingRetriever,
)

from _componentes import AvaliarRota, BuscaWeb, MontarContexto                  # noqa: E402

TEMPLATE_RESPOSTA = """
Voce e um assistente juridico especializado em controle externo (TCU).
Responda APENAS com base nos trechos abaixo, de forma objetiva. Se o trecho for da
WEB, deixe claro. Se nao constar, diga que nao consta.

Trechos:
{% for doc in documents %}
[{{ loop.index }}] ({{ doc.meta.get('tipo', 'local') }}) {{ doc.content }}
{% endfor %}

Pergunta: {{ question }}
Resposta:
"""

# Rotas do ConditionalRouter: so a query do ramo nao-local segue para o web search.
ROUTES = [
    {"condition": "{{rota == 'local'}}", "output": "{{question}}",
     "output_name": "q_local", "output_type": str},
    {"condition": "{{rota in ['fusao', 'web']}}", "output": "{{question}}",
     "output_name": "q_web", "output_type": str},
]


def _llm():
    groq_key, groq_modelo, groq_base = _comum.config_groq()
    return OpenAIGenerator(api_key=Secret.from_token(groq_key), model=groq_modelo,
                           api_base_url=groq_base,
                           generation_kwargs=_comum.gen_kwargs(groq_modelo, 0.2, 500))


def montar_pipeline(store, top_k, usar_langfuse):
    pipe = Pipeline()
    if usar_langfuse:
        from haystack_integrations.components.connectors.langfuse import LangfuseConnector

        pipe.add_component("tracer", LangfuseConnector("crag-aula8"))

    pipe.add_component("embedder", _comum.text_embedder())
    pipe.add_component("retriever", OpenSearchEmbeddingRetriever(document_store=store, top_k=top_k))
    pipe.add_component("avaliar", AvaliarRota())
    pipe.add_component("router", ConditionalRouter(routes=ROUTES))
    pipe.add_component("busca_web", BuscaWeb(max_results=3))
    pipe.add_component("montar", MontarContexto())
    pipe.add_component("prompt", PromptBuilder(template=TEMPLATE_RESPOSTA,
                                               required_variables=["documents", "question"]))
    pipe.add_component("llm", _llm())

    pipe.connect("embedder.embedding", "retriever.query_embedding")
    pipe.connect("retriever.documents", "avaliar.documents")
    pipe.connect("avaliar.rota", "router.rota")
    pipe.connect("avaliar.question", "router.question")
    pipe.connect("router.q_web", "busca_web.query")
    pipe.connect("avaliar.documents", "montar.documents_local")
    pipe.connect("avaliar.rota", "montar.rota")
    pipe.connect("busca_web.web_docs", "montar.web_docs")
    pipe.connect("montar.documents", "prompt.documents")
    pipe.connect("avaliar.question", "prompt.question")
    pipe.connect("prompt.prompt", "llm.prompt")
    return pipe


def responder(pipe, pergunta, usar_langfuse):
    saidas = {"avaliar", "montar"}
    if usar_langfuse:
        saidas.add("tracer")
    resultado = pipe.run(
        {"embedder": {"text": pergunta}, "avaliar": {"question": pergunta}},
        include_outputs_from=saidas)
    return resultado


def _rotulo_rota(rota):
    return "so local" if rota == "local" else "local+web" if rota == "fusao" else "so web"


def main():
    parser = argparse.ArgumentParser(description="CRAG com ConditionalRouter (Aula 8).")
    parser.add_argument("--pergunta", default=None)
    parser.add_argument("--indice", default=_comum.INDICE_TCU)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--sem-langfuse", action="store_true")
    parser.add_argument("--demo", action="store_true",
                        help="roda as queries curadas (local vs web) e mostra a rota de cada")
    args = parser.parse_args()

    usar_langfuse = _comum.langfuse_configurado() and not args.sem_langfuse

    print("=" * 60)
    print("  CRAG (Corrective RAG) com ConditionalRouter - Aula 8")
    print("=" * 60)
    print(f"LangFuse: {'ligado' if usar_langfuse else 'desligado'} | "
          f"Tavily: {'real' if _comum.tavily_configurado() else 'OFFLINE (stub)'}")
    print(f"Limiares: alto>={_comum.LIMITE_ALTO} | baixo>={_comum.LIMITE_BAIXO}")

    store = _comum.abrir_store(args.indice)
    try:
        if store.count_documents() == 0:
            print("[ATENCAO] Indice vazio. Rode antes: python 01_indexar_opensearch.py")
            return
    except Exception as e:
        print(f"[ATENCAO] nao consegui acessar o indice: {e}")
        return

    pipe = montar_pipeline(store, args.top_k, usar_langfuse)

    # Modo demo: roda o conjunto curado (queries 'local' e 'web') e compara as rotas.
    if args.demo:
        queries = _comum.carregar_queries_teste()
        if not queries:
            print("\n[ATENCAO] queries_crag_tcu.json nao encontrado em aula8/datasets.")
            return
        print(f"\nDEMO - {len(queries)} queries curadas (rota_esperada vs rota_obtida):\n")
        for q in queries:
            r = responder(pipe, q["pergunta"], usar_langfuse)
            rota = r["avaliar"]["rota"]
            esperada = q.get("rota_esperada", "?")
            marca = "OK" if (esperada == rota or (esperada == "web" and rota in ("web", "fusao"))) else "~"
            print(f"  [{marca}] {q['id']}: esperada={esperada:<5} obtida={rota:<6} "
                  f"score={r['avaliar']['score']:.2f} | {q['pergunta'][:60]}")
        print("\n(esperada=web aceita rota web OU fusao; o limiar pode variar com o corpus.)")
        return

    pergunta = args.pergunta or "quando as contas sao julgadas irregulares?"
    print(f"Pergunta: {pergunta}\n")
    resultado = responder(pipe, pergunta, usar_langfuse)

    score = resultado["avaliar"]["score"]
    rota = resultado["avaliar"]["rota"]
    docs: List[Document] = resultado["montar"]["documents"]
    resposta = resultado["llm"]["replies"][0]

    print(f"Score medio dos documentos locais: {score:.2f}")
    print(f"ROTA escolhida: {rota.upper()} ({_rotulo_rota(rota)})")
    print(f"Fontes usadas: {[d.meta.get('id_original') for d in docs]}\n")
    print(f"Resposta:\n{resposta}")
    if usar_langfuse:
        url = resultado.get("tracer", {}).get("trace_url", "")
        if url:
            print(f"\nTrace LangFuse: {url}")


if __name__ == "__main__":
    main()
