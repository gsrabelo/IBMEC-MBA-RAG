"""
01_gerar_perguntas.py - Gera perguntas de avaliacao A PARTIR dos acordaos do TCU.

Por que? As perguntas que vem com a aula sao de direito penal e NAO casam com o
indice do TCU (aula4_hibrido). Para a avaliacao ficar coerente e facil de explicar,
aqui a gente GERA as perguntas a partir dos proprios acordaos: para cada documento,
o LLM (Groq) cria uma pergunta objetiva + a resposta correta (ground_truth) tirada
do texto. Assim a resposta SEMPRE existe no corpus.

Saida: perguntas_geradas.json  ->  consumida automaticamente pelo 02_gerar_dataset.py.

Precisa da chave da Groq. Le os acordaos de ../../aula4/datasets.

Uso:
    python 01_gerar_perguntas.py
    python 01_gerar_perguntas.py --limite 10
"""

import argparse
import json
import re

import _comum

PROMPT = (
    "Voce recebe um ACORDAO do TCU. Gere UMA pergunta objetiva, clara e especifica "
    "que possa ser respondida SOMENTE com base neste acordao, e a resposta correta "
    "(curta e factual, extraida do texto). Evite perguntas genericas. "
    'Responda em JSON valido, exatamente neste formato: '
    '{{"pergunta": "...", "resposta": "..."}}\n\nACORDAO:\n{texto}'
)


def extrair_json(texto):
    """Tenta interpretar a saida do LLM como JSON {pergunta, resposta}."""
    try:
        return json.loads(texto)
    except Exception:
        pass
    # fallback: pega o primeiro bloco { ... } do texto
    m = re.search(r"\{.*\}", texto, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return None
    return None


def main():
    parser = argparse.ArgumentParser(description="Gera perguntas a partir dos acordaos do TCU.")
    parser.add_argument("--limite", type=int, default=10, help="quantos acordaos usar (1 pergunta cada)")
    parser.add_argument("--saida", default=str(_comum.PERGUNTAS_GERADAS), help="arquivo de saida JSON")
    args = parser.parse_args()

    _comum.carregar_env()
    cliente, modelo = _comum.groq_client()
    acordaos = _comum.carregar_acordaos(limite=args.limite)

    print("=" * 60)
    print("  GERAR PERGUNTAS DOS ACORDAOS - Aula 5")
    print("=" * 60)
    print(f"Acordaos: {len(acordaos)} | LLM: {modelo}")

    perguntas = []
    for n, doc in enumerate(acordaos, 1):
        try:
            resp = cliente.chat.completions.create(
                model=modelo,
                messages=[{"role": "user", "content": PROMPT.format(texto=doc["texto"][:2500])}],
                temperature=0.3, max_tokens=350,
            )
            par = extrair_json(resp.choices[0].message.content)
        except Exception as e:
            par = None
            if n == 1:
                print(f"  (aviso: falha ao chamar a Groq - {e})")

        if not par or "pergunta" not in par or "resposta" not in par:
            print(f"  [{n}/{len(acordaos)}] (pulado: o LLM nao retornou JSON valido)")
            continue

        perguntas.append({
            "id": f"GEN{n:03d}",
            "question": str(par["pergunta"]).strip(),
            "ground_truth": str(par["resposta"]).strip(),
            "tipo": "acordao_tcu",
            "doc_id": doc.get("id"),
        })
        print(f"  [{n}/{len(acordaos)}] {perguntas[-1]['question'][:70]}")

    with open(args.saida, "w", encoding="utf-8") as f:
        json.dump(perguntas, f, ensure_ascii=False, indent=2)
    print(f"\nPronto! {len(perguntas)} perguntas salvas em: {args.saida}")
    print("Agora rode: python 02_gerar_dataset.py --indice aula4_hibrido")


if __name__ == "__main__":
    main()
