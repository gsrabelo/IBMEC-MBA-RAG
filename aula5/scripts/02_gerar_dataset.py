"""
02_gerar_dataset.py - Gera o dataset de avaliacao (contexts + answer).

As perguntas (de 01_gerar_perguntas.py, ou as originais da aula) so tem a pergunta
e o ground_truth. Para avaliar um RAG (RAGAS/DeepEval) a gente precisa TAMBEM dos
trechos recuperados (contexts) e da resposta gerada (answer).

Este script:
  1. BUSCA (hibrida: BM25 + densa + RRF) os trechos no indice -> contexts
  2. GERA a resposta com a Groq usando esses trechos -> answer
  (busca e geracao sao SEPARADAS: se a Groq falhar, os contexts sao preservados)

Para cada pergunta salva: {question, contexts, answer, ground_truth, tipo}
A saida (dataset_avaliacao_completo.json) e consumida por 03/04/05.

Precisa de OpenSearch (com o indice indexado), Ollama e Groq.

Uso:
    python 02_gerar_dataset.py
    python 02_gerar_dataset.py --indice aula4_hibrido --top-k 5
    python 02_gerar_dataset.py --perguntas ..\datasets\corpus_avaliacao_aula5.json
"""

import argparse
import json

import _comum

_comum.carregar_env()  # antes de importar haystack

from haystack import Pipeline                                                 # noqa: E402
from haystack.components.builders import PromptBuilder                         # noqa: E402
from haystack.components.generators import OpenAIGenerator                     # noqa: E402
from haystack.components.joiners import DocumentJoiner                         # noqa: E402
from haystack.utils import Secret                                             # noqa: E402
from haystack_integrations.components.embedders.ollama import OllamaTextEmbedder  # noqa: E402
from haystack_integrations.components.retrievers.opensearch import (          # noqa: E402
    OpenSearchBM25Retriever,
    OpenSearchEmbeddingRetriever,
)
from haystack_integrations.document_stores.opensearch import OpenSearchDocumentStore  # noqa: E402

TEMPLATE = """
Voce e um assistente juridico. Responda APENAS com base nos trechos abaixo.
Se a informacao nao estiver neles, diga que nao consta. Seja objetivo.

Trechos:
{% for doc in documents %}
- {{ doc.content }}
{% endfor %}

Pergunta: {{ question }}
Resposta:
"""


def abrir_store(indice):
    base_url, modelo = _comum.config_ollama()
    os_cfg = _comum.config_opensearch()
    auth = (os_cfg["usuario"], os_cfg["senha"]) if os_cfg["usuario"] else None
    store = OpenSearchDocumentStore(
        hosts=os_cfg["url"], index=indice, embedding_dim=_comum.dimensao_do_modelo(modelo),
        http_auth=auth, use_ssl=False, verify_certs=False,
    )
    return store, base_url, modelo


def montar_busca(store, base_url, modelo, top_k):
    """Pipeline SO de busca (sem LLM): embedder + bm25 + densa -> RRF."""
    pipe = Pipeline()
    pipe.add_component("embedder", OllamaTextEmbedder(model=modelo, url=base_url))
    pipe.add_component("bm25", OpenSearchBM25Retriever(document_store=store, top_k=top_k))
    pipe.add_component("denso", OpenSearchEmbeddingRetriever(document_store=store, top_k=top_k))
    pipe.add_component("juntar", DocumentJoiner(join_mode="reciprocal_rank_fusion", top_k=top_k))
    pipe.connect("embedder.embedding", "denso.query_embedding")
    pipe.connect("bm25.documents", "juntar.documents")
    pipe.connect("denso.documents", "juntar.documents")
    return pipe


def montar_geracao():
    """Gerador isolado (Groq) com um PromptBuilder."""
    groq_key, groq_modelo, groq_base = _comum.config_groq()
    pipe = Pipeline()
    pipe.add_component("prompt", PromptBuilder(template=TEMPLATE, required_variables=["documents", "question"]))
    pipe.add_component("llm", OpenAIGenerator(
        api_key=Secret.from_token(groq_key), model=groq_modelo, api_base_url=groq_base,
        generation_kwargs={"temperature": 0.2, "max_tokens": 500}))
    pipe.connect("prompt.prompt", "llm.prompt")
    return pipe


def main():
    parser = argparse.ArgumentParser(description="Gera o dataset de avaliacao (contexts+answer).")
    parser.add_argument("--indice", default="aula4_hibrido", help="indice a avaliar (RAG)")
    parser.add_argument("--limite", type=int, default=10, help="quantas perguntas usar (0 = todas)")
    parser.add_argument("--top-k", type=int, default=5, help="quantos trechos recuperar")
    parser.add_argument("--saida", default=str(_comum.DATASET_COMPLETO), help="arquivo de saida JSON")
    parser.add_argument("--perguntas", default=None,
                        help="arquivo de perguntas (padrao: perguntas_geradas.json do 01)")
    args = parser.parse_args()

    perguntas = _comum.carregar_perguntas(caminho=args.perguntas, limite=args.limite)
    print("=" * 60)
    print("  GERAR DATASET DE AVALIACAO - Aula 5")
    print("=" * 60)
    print(f"Indice: {args.indice} | Perguntas: {len(perguntas)} | top-k: {args.top_k}")

    store, base_url, modelo = abrir_store(args.indice)

    # Diagnostico: o indice tem documentos? (causa comum de "0 trechos")
    try:
        total = store.count_documents()
        print(f"Documentos no indice '{args.indice}': {total}")
        if total == 0:
            print("\n[ATENCAO] O indice esta VAZIO. Rode antes o indexador da Aula 4:")
            print("  cd ../../aula4/scripts && python 01_indexar_hibrido.py --recriar")
            return
    except Exception as e:
        print(f"[ATENCAO] Nao consegui contar documentos ({e}). O indice existe?")
        return

    busca = montar_busca(store, base_url, modelo, args.top_k)
    geracao = montar_geracao()

    dataset = []
    erros_busca = erros_llm = 0
    for n, p in enumerate(perguntas, 1):
        # 1) BUSCA (preserva os contexts mesmo se a geracao falhar depois)
        try:
            docs = busca.run(
                {"embedder": {"text": p["question"]}, "bm25": {"query": p["question"]}}
            )["juntar"]["documents"]
            contexts = [d.content for d in docs]
        except Exception as e:
            docs, contexts = [], []
            erros_busca += 1
            if erros_busca == 1:
                print(f"  [erro de BUSCA] {type(e).__name__}: {str(e)[:120]}")

        # 2) GERACAO (so se houver contexts)
        answer = ""
        if contexts:
            try:
                answer = geracao.run(
                    {"prompt": {"documents": docs, "question": p["question"]}}
                )["llm"]["replies"][0]
            except Exception as e:
                answer = f"[ERRO LLM: {e}]"
                erros_llm += 1
                if erros_llm == 1:
                    print(f"  [erro de LLM] {type(e).__name__}: {str(e)[:120]}")

        dataset.append({
            "id": p.get("id"), "question": p["question"], "contexts": contexts,
            "answer": answer, "ground_truth": p.get("ground_truth", ""), "tipo": p.get("tipo", ""),
        })
        print(f"  [{n}/{len(perguntas)}] {p['question'][:50]}... -> {len(contexts)} trechos")

    with open(args.saida, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    print(f"\nPronto! {len(dataset)} pares salvos em: {args.saida}")
    if erros_busca:
        print(f"[!] {erros_busca} perguntas tiveram erro de BUSCA (veja a mensagem acima).")
    if erros_llm:
        print(f"[!] {erros_llm} perguntas tiveram erro de LLM (contexts foram preservados).")
    print("Agora rode: python 03_ragas_avaliar.py")


if __name__ == "__main__":
    main()
