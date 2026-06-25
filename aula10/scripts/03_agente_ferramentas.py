"""
03_agente_ferramentas.py - Agentic RAG com o Agent do Haystack + 3 ferramentas + LangFuse.

Usa o componente Agent do Haystack (laco ReAct/tool-calling nativo). O LLM (Groq) decide
sozinho quais ferramentas usar e em que ordem, ate responder:
  - buscar_documentos (jurisprudencia / OpenSearch)
  - buscar_web        (fatos recentes / Tavily)
  - consultar_banco   (dados estruturados / SQLite via text-to-SQL)

Com LangFuse ligado, o agente roda dentro de um Pipeline com LangfuseConnector e cada
passo (pensamento, chamada de ferramenta, resposta) e capturado no trace 'agente-aula10'.

Uso:
    python 03_agente_ferramentas.py --pergunta "quantas ocorrencias de estelionato em SP e o que a jurisprudencia diz?"
    python 03_agente_ferramentas.py --pergunta "..." --max-passos 5 --sem-langfuse
"""

import argparse

import _comum

_comum.carregar_env()  # antes de importar haystack (liga o tracing)

from haystack import Pipeline                                                 # noqa: E402
from haystack.dataclasses import ChatMessage                                  # noqa: E402


def montar_pipeline(max_passos, usar_langfuse):
    agente = _comum.criar_agente(max_passos=max_passos)
    pipe = Pipeline()
    if usar_langfuse:
        from haystack_integrations.components.connectors.langfuse import LangfuseConnector

        pipe.add_component("tracer", LangfuseConnector("agente-aula10"))
    pipe.add_component("agente", agente)
    return pipe


def resumir_trajetoria(mensagens):
    """Lista as ferramentas chamadas (a 'trajetoria' do agente) a partir das mensagens."""
    passos = []
    for m in mensagens:
        for tc in (getattr(m, "tool_calls", None) or []):
            nome = getattr(tc, "tool_name", None) or getattr(tc, "name", "?")
            args = getattr(tc, "arguments", None) or getattr(tc, "args", {})
            passos.append(f"{nome}({args})")
    return passos


SINAIS_INSATISFATORIA = (
    "nao consta", "não consta", "nao encontrei", "não encontrei", "nao foi possivel",
    "não foi possível", "nao ha informacao", "não há informação", "sem informacao",
    "sem informação", "nao localizei", "não localizei",
)

PROMPT_FALLBACK = (
    "A busca local nao trouxe base suficiente. Use os resultados da WEB abaixo para "
    "responder a pergunta de forma objetiva, deixando claro que a fonte e a WEB (menos "
    "autoritativa que a jurisprudencia local). Se a web tambem nao ajudar, diga que nao "
    "foi possivel confirmar.\n\nResultados da web:\n{contexto}\n\nPergunta: {pergunta}\nResposta:"
)


def resposta_insatisfatoria(texto):
    """Heuristica: a resposta indica que nao achou base (ou ficou curta demais)?"""
    if not texto or len(texto.strip()) < 40:
        return True
    baixo = texto.lower()
    return any(s in baixo for s in SINAIS_INSATISFATORIA)


def fallback_web(pergunta, resposta, traj):
    """Se a resposta veio fraca e o agente NAO usou a web, aciona buscar_web e refaz.

    Padrao corretivo (estilo CRAG): so dispara quando (1) a resposta parece insuficiente
    e (2) a web ainda nao foi tentada. Exige TAVILY_API_KEY (senao apenas avisa).
    """
    usou_web = any("buscar_web" in p for p in traj)
    if usou_web or not resposta_insatisfatoria(resposta):
        return resposta, False
    print("\n[fallback] resposta local insuficiente -> acionando web search (Tavily)...")
    if not _comum.tavily_configurado():
        print("[fallback] sem TAVILY_API_KEY - configure a chave para o fallback web funcionar.")
        return resposta, False
    contexto = _comum.buscar_web(pergunta)
    cliente, modelo = _comum.groq_client()
    nova = _comum.gerar_texto(cliente, modelo,
                              PROMPT_FALLBACK.format(contexto=contexto, pergunta=pergunta),
                              max_tokens=500, temperature=0.2)
    return nova, True


def main():
    parser = argparse.ArgumentParser(description="Agentic RAG com Haystack Agent (Aula 10).")
    parser.add_argument("--pergunta", required=True)
    parser.add_argument("--max-passos", type=int, default=5)
    parser.add_argument("--sem-langfuse", action="store_true")
    parser.add_argument("--sem-fallback-web", action="store_true",
                        help="desliga o fallback corretivo p/ web quando a resposta vem fraca")
    args = parser.parse_args()

    usar_langfuse = _comum.langfuse_configurado() and not args.sem_langfuse
    print("=" * 60)
    print("  AGENTIC RAG (Haystack Agent) - Aula 10")
    print("=" * 60)
    print(f"Pergunta: {args.pergunta}")
    print(f"LangFuse: {'ligado (trace agente-aula10)' if usar_langfuse else 'desligado'} | "
          f"max_passos={args.max_passos}\n")

    pipe = montar_pipeline(args.max_passos, usar_langfuse)
    saidas = {"agente"} | ({"tracer"} if usar_langfuse else set())
    resultado = pipe.run({"agente": {"messages": [ChatMessage.from_user(args.pergunta)]}},
                         include_outputs_from=saidas)

    saida = resultado["agente"]
    mensagens = saida.get("messages", [])
    traj = resumir_trajetoria(mensagens)
    print("Trajetoria (ferramentas chamadas):")
    for i, p in enumerate(traj, 1):
        print(f"  {i}. {p}")
    if not traj:
        print("  (nenhuma - o agente respondeu por conhecimento geral)")

    resposta = saida["last_message"].text if saida.get("last_message") else "(sem resposta)"

    # Fallback corretivo: se a resposta veio fraca e o agente nao usou a web, tenta a web.
    origem = "agente"
    if not args.sem_fallback_web:
        nova, acionou = fallback_web(args.pergunta, resposta, traj)
        if acionou:
            resposta, origem = nova, "fallback-web"

    print(f"\nResposta ({origem}):\n{resposta}")
    if usar_langfuse:
        url = resultado.get("tracer", {}).get("trace_url", "")
        if url:
            print(f"\nTrace LangFuse: {url}")


if __name__ == "__main__":
    main()
