"""
04_adaptive_rag.py - Adaptive RAG: classifica a complexidade e roteia em 3 caminhos.

Nem toda pergunta precisa do agente caro. Um CLASSIFICADOR (LLM) estima a complexidade
e roteia (mesma ideia do ConditionalRouter da Aula 8):
  - sem_retrieval : conhecimento geral basta            -> responde direto (barato/rapido)
  - simples       : factual, 1 busca resolve            -> 1x buscar_documentos + responder
  - complexa      : comparativa/multi-fonte             -> AGENTE (ReAct, varias ferramentas)

Imprime a rota escolhida e a resposta. Compare o custo/latencia entre as rotas.

Uso:
    python 04_adaptive_rag.py --pergunta "o que e habeas corpus?"
    python 04_adaptive_rag.py --pergunta "qual a pena do art. 2 da Lei 12.850?"
    python 04_adaptive_rag.py --pergunta "compare prisao preventiva em crimes financeiros vs violentos"
"""

import argparse
import time

import _comum

_comum.carregar_env()

PROMPT_CLASSIFICADOR = (
    "Classifique a COMPLEXIDADE da pergunta juridica para escolher a estrategia de RAG.\n"
    "Responda APENAS uma palavra: sem_retrieval | simples | complexa\n"
    "- sem_retrieval: conhecimento geral/definicao, nao precisa buscar (ex.: 'o que e habeas corpus?')\n"
    "- simples: factual, 1 busca resolve (ex.: 'qual o prazo do recurso de apelacao?')\n"
    "- complexa: comparar fontes/varios passos/fatos recentes (ex.: 'compare precedentes X vs Y de 2022-2024')\n\n"
    "Pergunta: {pergunta}\nClasse:"
)

PROMPT_DIRETO = ("Responda a pergunta juridica de forma objetiva, com seu conhecimento geral. "
                 "Se nao tiver certeza, diga que conviria consultar a fonte.\n\nPergunta: {pergunta}\nResposta:")

PROMPT_SIMPLES = ("Responda a pergunta usando APENAS os trechos abaixo, de forma objetiva. "
                  "Se nao constar, diga que nao consta.\n\nTrechos:\n{contexto}\n\nPergunta: {pergunta}\nResposta:")


def classificar(cliente, modelo, pergunta):
    r = _comum.gerar_texto(cliente, modelo, PROMPT_CLASSIFICADOR.format(pergunta=pergunta),
                           max_tokens=8, temperature=0.0).lower()
    for classe in ("sem_retrieval", "simples", "complexa"):
        if classe in r:
            return classe
    return "simples"  # default seguro


def rota_sem_retrieval(cliente, modelo, pergunta):
    return _comum.gerar_texto(cliente, modelo, PROMPT_DIRETO.format(pergunta=pergunta), max_tokens=500, temperature=0.2)


def rota_simples(cliente, modelo, pergunta):
    contexto = _comum.buscar_documentos(pergunta)
    return _comum.gerar_texto(cliente, modelo,
                              PROMPT_SIMPLES.format(contexto=contexto, pergunta=pergunta),
                              max_tokens=500, temperature=0.2)


def rota_complexa(pergunta, max_passos):
    from haystack.dataclasses import ChatMessage

    agente = _comum.criar_agente(max_passos=max_passos)
    saida = agente.run(messages=[ChatMessage.from_user(pergunta)])
    return saida["last_message"].text if saida.get("last_message") else "(sem resposta)"


def responder_adaptativo(pergunta, max_passos=5):
    cliente, modelo = _comum.groq_client()
    classe = classificar(cliente, modelo, pergunta)
    t0 = time.perf_counter()
    if classe == "sem_retrieval":
        resp = rota_sem_retrieval(cliente, modelo, pergunta)
    elif classe == "complexa":
        resp = rota_complexa(pergunta, max_passos)
    else:
        resp = rota_simples(cliente, modelo, pergunta)
    return classe, resp, time.perf_counter() - t0


def main():
    parser = argparse.ArgumentParser(description="Adaptive RAG - 3 caminhos (Aula 10).")
    parser.add_argument("--pergunta", required=True)
    parser.add_argument("--max-passos", type=int, default=5)
    args = parser.parse_args()

    print("=" * 60)
    print("  ADAPTIVE RAG (classificador + 3 caminhos) - Aula 10")
    print("=" * 60)
    print(f"Pergunta: {args.pergunta}")
    classe, resp, dt = responder_adaptativo(args.pergunta, args.max_passos)
    rotulo = {"sem_retrieval": "SEM RETRIEVAL (direto)",
              "simples": "SINGLE-STEP RAG (1 busca)",
              "complexa": "MULTI-STEP (agente)"}[classe]
    print(f"\nClasse: {classe}  ->  Rota: {rotulo}  ({dt:.1f}s)")
    print(f"\nResposta:\n{resp}")


if __name__ == "__main__":
    main()
