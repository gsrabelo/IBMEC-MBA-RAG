"""
01_query_rewriting.py - Reescrita de query (Advanced RAG).

Perguntas em linguagem coloquial costumam "errar" na busca porque os documentos
usam termos tecnicos. Query rewriting reescreve a pergunta para aproximar do corpus.
Tres tecnicas (todas usando o LLM da Groq):

  1. paraphrase - reescreve com terminologia juridica tecnica
  2. hyde       - gera um paragrafo hipotetico que responderia a pergunta (HyDE-Lite)
  3. stepback   - generaliza a pergunta para o conceito por tras dela

Precisa da chave da Groq no .env. Nao precisa de OpenSearch nem Ollama.

Uso:
    python 01_query_rewriting.py
    python 01_query_rewriting.py --query "Podem prender alguem sem mandado?"
    python 01_query_rewriting.py --tecnica hyde
"""

import argparse

import _comum

QUERY_EXEMPLO = "Podem prender alguem sem mandado?"


def main():
    parser = argparse.ArgumentParser(description="Reescrita de query para Advanced RAG.")
    parser.add_argument("--query", default=QUERY_EXEMPLO, help="pergunta original")
    parser.add_argument("--tecnica", default="todas",
                        choices=["paraphrase", "hyde", "stepback", "todas"],
                        help="qual tecnica usar (padrao: todas)")
    args = parser.parse_args()

    _comum.carregar_env()
    cliente, modelo = _comum.groq_client()

    print("=" * 60)
    print(f"  QUERY REWRITING - Aula 3 (LLM: {modelo})")
    print("=" * 60)
    print(f"Query original: {args.query}")

    tecnicas = ["paraphrase", "hyde", "stepback"] if args.tecnica == "todas" else [args.tecnica]
    rotulos = {"paraphrase": "PARAPHRASE (termos tecnicos)",
               "hyde": "HyDE-LITE (documento hipotetico)",
               "stepback": "STEP-BACK (pergunta mais geral)"}

    for t in tecnicas:
        print(f"\n[{rotulos[t]}]")
        print("-" * 50)
        try:
            print(_comum.reescrever_query(cliente, modelo, args.query, t))
        except Exception as e:
            print(f"(falhou - verifique a chave da Groq: {e})")

    print("\nDica: no Advanced RAG, a query reescrita e usada na BUSCA; "
          "a resposta final ainda responde a pergunta original.")


if __name__ == "__main__":
    main()
