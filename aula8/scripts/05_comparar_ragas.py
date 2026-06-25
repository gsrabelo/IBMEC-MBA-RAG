"""
05_comparar_ragas.py - CRAG vs Advanced RAG avaliado com RAGAS (Faithfulness).

Objetivo da aula: medir se a AUTO-CORRECAO do CRAG melhora a Faithfulness (fidelidade
factual) em relacao a um RAG "avancado" simples (busca densa + geracao, sem correcao).

Para cada pergunta de teste:
  - Advanced RAG : busca densa top-k -> gera resposta
  - CRAG         : recupera -> avalia -> roteia (local/fusao/web) -> gera resposta
Depois roda RAGAS (Faithfulness + ResponseRelevancy) nas duas e compara.

O conjunto de teste e gerado a partir do PROPRIO indice do TCU (perguntas coloquiais
cujo gabarito e o acordao de origem), garantindo que as respostas existem no corpus.

Uso:
    python 05_comparar_ragas.py                 # 6 perguntas geradas do indice
    python 05_comparar_ragas.py --n 8 --top-k 4
"""

import argparse
import random

import _comum

_comum.carregar_env()


def gerar_perguntas(cliente, modelo, store, n):
    """Gera N perguntas coloquiais a partir de documentos que ESTAO no indice."""
    docs = store.filter_documents()
    random.seed(42)
    random.shuffle(docs)
    perguntas = []
    for d in docs[:n]:
        prompt = ("Leia o trecho juridico e gere UMA pergunta objetiva, coloquial, "
                  "respondivel SO com ele. Responda apenas a pergunta.\n\n"
                  f"Trecho: {d.content[:1200]}")
        pergunta = _comum.gerar_texto(cliente, modelo, prompt, max_tokens=80, temperature=0.5)
        perguntas.append({"pergunta": pergunta.strip(), "gabarito": d.content})
    return perguntas


def resposta_advanced(cliente, modelo, pipe_busca, pergunta, top_k):
    """Advanced RAG: busca densa + geracao (sem correcao)."""
    docs = _comum.buscar(pipe_busca, pergunta)
    resposta = _comum.responder_com_contexto(cliente, modelo, pergunta, docs)
    return resposta, [d.content for d in docs]


def resposta_crag(modulo_crag, pipe_crag, pergunta, usar_langfuse):
    """CRAG: usa o pipeline do 04 (retrieve -> avaliar -> rota -> gerar)."""
    resultado = modulo_crag.responder(pipe_crag, pergunta, usar_langfuse)
    docs = resultado["montar"]["documents"]
    resposta = resultado["llm"]["replies"][0]
    return resposta, [d.content for d in docs], resultado["avaliar"]["rota"]


def avaliar_ragas(amostras):
    """Roda RAGAS (Faithfulness + ResponseRelevancy) numa lista de amostras."""
    from langchain_groq import ChatGroq
    from ragas import EvaluationDataset, evaluate
    from ragas.dataset_schema import SingleTurnSample
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import Faithfulness, ResponseRelevancy

    groq_key, groq_modelo, _ = _comum.config_groq()
    juiz = LangchainLLMWrapper(ChatGroq(model=groq_modelo, api_key=groq_key, temperature=0))

    base_url, modelo_emb = _comum.config_ollama()
    try:
        from langchain_ollama import OllamaEmbeddings
    except ImportError:
        from langchain_community.embeddings import OllamaEmbeddings
    emb = LangchainEmbeddingsWrapper(OllamaEmbeddings(model=modelo_emb, base_url=base_url))

    samples = [SingleTurnSample(user_input=a["pergunta"], response=a["resposta"],
                                retrieved_contexts=a["contextos"]) for a in amostras]
    dataset = EvaluationDataset(samples=samples)
    # strictness=1: a ResponseRelevancy gera 1 pergunta por amostra (n=1). A Groq NAO
    # aceita n>1 ("'n' must be at most 1"); o padrao 3 do RAGAS quebra os jobs.
    metricas = [Faithfulness(), ResponseRelevancy(strictness=1)]
    resultado = evaluate(dataset=dataset, metrics=metricas, llm=juiz, embeddings=emb)
    return resultado.to_pandas()


def main():
    parser = argparse.ArgumentParser(description="CRAG vs Advanced RAG com RAGAS (Aula 8).")
    parser.add_argument("--indice", default=_comum.INDICE_TCU)
    parser.add_argument("--n", type=int, default=6, help="nº de perguntas de teste")
    parser.add_argument("--top-k", type=int, default=4)
    args = parser.parse_args()

    print("=" * 60)
    print("  CRAG vs ADVANCED RAG - avaliacao RAGAS (Aula 8)")
    print("=" * 60)

    cliente, modelo = _comum.groq_client()
    store = _comum.abrir_store(args.indice)
    if store.count_documents() == 0:
        print("[ATENCAO] Indice vazio. Rode antes: python 01_indexar_opensearch.py")
        return

    print(f"Gerando {args.n} perguntas de teste a partir do indice...")
    perguntas = gerar_perguntas(cliente, modelo, store, args.n)

    pipe_busca = _comum.montar_busca(store, args.top_k)
    modulo_crag = _comum.importar_script("04_crag.py")
    usar_langfuse = _comum.langfuse_configurado()
    pipe_crag = modulo_crag.montar_pipeline(store, args.top_k, usar_langfuse)

    amostras_adv, amostras_crag = [], []
    for i, item in enumerate(perguntas, 1):
        q = item["pergunta"]
        print(f"\n[{i}/{len(perguntas)}] {q}")
        r_adv, ctx_adv = resposta_advanced(cliente, modelo, pipe_busca, q, args.top_k)
        r_crag, ctx_crag, rota = resposta_crag(modulo_crag, pipe_crag, q, usar_langfuse)
        print(f"   Advanced: {r_adv[:90]}...")
        print(f"   CRAG (rota={rota}): {r_crag[:90]}...")
        amostras_adv.append({"pergunta": q, "resposta": r_adv, "contextos": ctx_adv})
        amostras_crag.append({"pergunta": q, "resposta": r_crag, "contextos": ctx_crag})

    print("\nRodando RAGAS (pode demorar)...")
    df_adv = avaliar_ragas(amostras_adv)
    df_crag = avaliar_ragas(amostras_crag)

    def media(df, col):
        return float(df[col].mean()) if col in df.columns else float("nan")

    print("\n" + "=" * 60)
    print("  RESULTADO COMPARATIVO (media)")
    print("=" * 60)
    print(f"{'Metrica':<22}{'Advanced RAG':>15}{'CRAG':>10}")
    for col, nome in [("faithfulness", "Faithfulness"), ("answer_relevancy", "ResponseRelevancy")]:
        print(f"{nome:<22}{media(df_adv, col):>15.3f}{media(df_crag, col):>10.3f}")
    print("\nNota: se as duas ficarem proximas, e porque o indice do TCU ja responde "
          "bem as perguntas locais; o ganho do CRAG aparece quando o retrieval local "
          "falha (ai a rota web/fusao corrige).")


if __name__ == "__main__":
    main()
