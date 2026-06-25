"""
02_react_manual.py - O ciclo ReAct (Thought -> Action -> Observation) na unha.

Antes de usar o Agent do Haystack (script 03), implementamos o laco ReAct manualmente
para o aluno VER o ciclo: a cada passo o LLM (Groq) decide um pensamento e uma acao
(qual ferramenta + argumento); executamos a ferramenta e devolvemos a observacao; repete
ate o LLM decidir 'finalizar'. Ha um limite de passos (guard contra loop).

Ferramentas: buscar_documentos, buscar_web, consultar_banco (as mesmas do agente).

Uso:
    python 02_react_manual.py --pergunta "quantas ocorrencias de estelionato em SP e o que diz a lei sobre isso?"
    python 02_react_manual.py --pergunta "o que e habeas corpus?" --max-passos 4
"""

import argparse
import json
import re

import _comum

_comum.carregar_env()

FERRAMENTAS = {
    "buscar_documentos": _comum.buscar_documentos, # FERRAMENTA 1: buscar_documentos (jurisprudencia, OpenSearch)
    "buscar_web": _comum.buscar_web, # FERRAMENTA 2: buscar_web (fatos recentes)
    "consultar_banco": _comum.consultar_banco, # FERRAMENTA 3: consultar_banco (dados estruturados)
}

PROMPT_REACT = """Voce e um assistente juridico que resolve a pergunta em PASSOS (padrao ReAct).
A cada passo, responda APENAS um JSON:
  {{"thought": "...", "action": "buscar_documentos|buscar_web|consultar_banco|finalizar", "action_input": "...", "answer": "..."}}
- Use 'action' para chamar UMA ferramenta (action_input = argumento em texto).
- Quando tiver informacao suficiente, use action="finalizar" e preencha "answer".
Ferramentas:
  - buscar_documentos: jurisprudencia/acordaos no acervo local (vetorial).
  - buscar_web: fatos recentes na web.
  - consultar_banco: dados estruturados (acordaos, ocorrencias, legislacao, doutrina) via linguagem natural.

Pergunta: {pergunta}

Historico ate agora:
{historico}

Proximo passo (apenas o JSON):"""


def extrair_json(txt):
    m = re.search(r"\{.*\}", txt, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(description="Ciclo ReAct manual (Aula 10).")
    parser.add_argument("--pergunta", required=True)
    parser.add_argument("--max-passos", type=int, default=5)
    args = parser.parse_args()

    cliente, modelo = _comum.groq_client()
    historico = ""
    print("=" * 60)
    print("  CICLO ReAct MANUAL - Aula 10")
    print("=" * 60)
    print(f"Pergunta: {args.pergunta}\n")

    for passo in range(1, args.max_passos + 1):
        prompt = PROMPT_REACT.format(pergunta=args.pergunta, historico=historico or "(vazio)")
        bruto = _comum.gerar_texto(cliente, modelo, prompt, max_tokens=400, temperature=0.0)
        dados = extrair_json(bruto) or {"action": "finalizar", "answer": bruto}

        thought = dados.get("thought", "")
        acao = dados.get("action", "finalizar")
        print(f"--- Passo {passo} ---")
        if thought:
            print(f"Thought: {thought}")

        if acao == "finalizar" or acao not in FERRAMENTAS:
            print(f"\nAnswer: {dados.get('answer', bruto)}")
            return

        entrada = dados.get("action_input", "")
        print(f"Action: {acao}({entrada!r})")
        obs = FERRAMENTAS[acao](entrada)
        print(f"Observation: {obs[:800]}{'...' if len(obs) > 300 else ''}\n")
        historico += (f"\nThought: {thought}\nAction: {acao}({entrada})\nObservation: {obs[:600]}\n")

    print(f"\n[guard] limite de {args.max_passos} passos atingido sem 'finalizar'.")


if __name__ == "__main__":
    main()
