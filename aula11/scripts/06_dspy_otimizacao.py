"""
06_dspy_otimizacao.py - DSPy: otimizacao automatica do prompt do RAG.

Em vez de ajustar o prompt na mao, declaramos um modulo (ChainOfThought), uma metrica e
um pequeno dataset; o otimizador BootstrapFewShot escolhe automaticamente instrucoes e
exemplos few-shot que maximizam a metrica. Comparamos a resposta ANTES e DEPOIS da
compilacao.

LLM = Groq (via dspy.LM OpenAI-compatible). O 'gold' do treino e gerado uma vez pela
propria Groq (didatico: mostra a MECANICA da otimizacao, nao um benchmark rigoroso).

Precisa: pip install dspy-ai   e o indice: python 01_indexar.py

Uso:
    python 06_dspy_otimizacao.py
    python 06_dspy_otimizacao.py --n-treino 5 --pergunta "o que e roubo segundo o CP?"
"""

import argparse

import _comum

_comum.carregar_env()


def contexto_de(store, pergunta, top_k=3):
    docs = _comum.buscar(_comum.montar_busca(store, top_k), pergunta)
    return "\n".join(f"- {d.content}" for d in docs)


def main():
    parser = argparse.ArgumentParser(description="DSPy - otimizacao de prompt (Aula 11).")
    parser.add_argument("--n-treino", type=int, default=5, help="nº de exemplos de treino")
    parser.add_argument("--pergunta", default="Qual a pena prevista para o crime de roubo?")
    args = parser.parse_args()

    print("=" * 60)
    print("  DSPy - OTIMIZACAO AUTOMATICA DE PROMPT - Aula 11")
    print("=" * 60)

    try:
        import dspy
    except ImportError:
        print("[ERRO] instale: pip install dspy-ai")
        return

    store = _comum.abrir_store()
    if store.count_documents() == 0:
        print("[ATENCAO] indice vazio. Rode antes: python 01_indexar.py")
        return

    api_key, modelo, base_url = _comum.config_groq()
    lm = dspy.LM(f"openai/{modelo}", api_base=base_url, api_key=api_key, max_tokens=600, temperature=0.2)
    dspy.configure(lm=lm)

    # modulo RAG: contexto + pergunta -> resposta (com raciocinio)
    modulo = dspy.ChainOfThought("context, question -> answer")

    # dataset de treino: pega queries do benchmark, recupera contexto e gera um 'gold'
    cliente, gmodelo = _comum.groq_client()
    queries = _comum.carregar_queries()[: args.n_treino]
    trainset = []
    print(f"Montando trainset com {len(queries)} exemplos (gera gold via Groq)...")
    for q in queries:
        pergunta = q["query"]
        ctx = contexto_de(store, pergunta)
        gold = _comum.responder_com_contexto(cliente, gmodelo, pergunta, [ctx])
        trainset.append(dspy.Example(context=ctx, question=pergunta, answer=gold)
                        .with_inputs("context", "question"))

    # metrica simples: sobreposicao de palavras entre resposta e gold (>= 0.3)
    def metrica(exemplo, pred, trace=None):
        a = set((pred.answer or "").lower().split())
        b = set((exemplo.answer or "").lower().split())
        if not b:
            return False
        return (len(a & b) / len(b)) >= 0.3

    pergunta_teste = args.pergunta
    ctx_teste = contexto_de(store, pergunta_teste)

    print("\n--- ANTES (sem otimizacao) ---")
    antes = modulo(context=ctx_teste, question=pergunta_teste)
    print(antes.answer[:400])

    print("\nCompilando com BootstrapFewShot (escolhe few-shot automaticamente)...")
    otim = dspy.BootstrapFewShot(metric=metrica, max_bootstrapped_demos=2, max_labeled_demos=2)
    compilado = otim.compile(modulo, trainset=trainset)

    print("\n--- DEPOIS (otimizado) ---")
    depois = compilado(context=ctx_teste, question=pergunta_teste)
    print(depois.answer[:400])

    # mostra quantos exemplos few-shot foram embutidos no prompt
    try:
        demos = compilado.predictors()[0].demos
        print(f"\nFew-shot embutidos pelo DSPy: {len(demos)}")
    except Exception:
        pass
    print("\nLeitura: o DSPy injeta instrucoes/exemplos escolhidos por bootstrapping. "
          "Use dspy.inspect_history(n=1) para ver o prompt final compilado.")

    print(dspy.inspect_history(n=1))


if __name__ == "__main__":
    main()
