"""
05_benchmark.py - Trade-off: Recall vs Latencia vs Custo (Aula 7).

Compara a BUSCA (so recuperacao, sem gerar resposta) de 4 estrategias sobre o
indice do TCU:
  - Baseline   : 1 busca com a pergunta crua
  - Multi-Query: variacoes -> busca cada -> dedup (ordenado por score)
  - Step-Back  : pergunta + versao geral -> busca -> dedup
  - RAG-Fusion : variacoes -> busca cada -> RRF

Mede: Hit@k, Recall@k (usando o gabarito), latencia media e custo (nº de buscas e
de chamadas LLM por pergunta).

DOIS modos de benchmark (gerados a partir dos docs QUE ESTAO NO INDICE):
  - padrao    : perguntas COLOQUIAIS de 1 documento (gabarito = 1 doc). Cenario de
                "lookup" - o Baseline costuma ir bem e o enhancement nao compensa.
  - --multi-doc: perguntas TEMATICAS com VARIOS documentos relevantes (via clustering).
                Cenario em que Multi-Query/RAG-Fusion ganham (recuperam mais relevantes).

Precisa de OpenSearch (indice do TCU populado), Ollama e Groq.

Uso:
    python 05_benchmark.py
    python 05_benchmark.py --multi-doc --top-k 10
    python 05_benchmark.py --gerar          # forca regerar o benchmark do modo atual
"""

import argparse
import json
import random
import re
import time

import numpy as np
from sklearn.cluster import KMeans

import _comum

BENCHMARK_MULTIDOC = _comum.PASTA_DATASETS / "benchmark_multidoc.json"

PROMPT_COLOQUIAL = (
    "Com base no acordao do TCU abaixo, gere UMA pergunta em LINGUAGEM COLOQUIAL/leiga "
    "(como um cidadao comum perguntaria, SEM jargao juridico) cuja resposta esteja no "
    "documento, e a resposta correta em linguagem tecnica. "
    'Responda em JSON: {{"pergunta": "...", "resposta": "..."}}\n\n{texto}'
)
PROMPT_TEMATICA = (
    "Os trechos abaixo sao de varios acordaos do TCU sobre um MESMO tema. Gere UMA "
    "pergunta que um CIDADAO LEIGO faria sobre esse tema, em LINGUAGEM COLOQUIAL, e uma "
    "resposta que resume o panorama. REGRAS IMPORTANTES:\n"
    "- NAO cite numeros de acordao (ex.: 1234/2026) nem nomes de orgaos especificos.\n"
    "- NAO copie os termos tecnicos/juridicos dos trechos; use palavras do dia a dia.\n"
    "- A pergunta deve ser sobre o TEMA geral (respondivel olhando varios casos), nao "
    "sobre um caso especifico.\n"
    'Responda em JSON: {{"pergunta": "...", "resposta": "..."}}\n\n{trechos}'
)


def extrair_json(texto):
    if not texto:
        return None
    # remove cercas de codigo (```json ... ```) que alguns modelos adicionam
    texto = re.sub(r"```(?:json)?", "", texto).strip()
    try:
        return json.loads(texto)
    except Exception:
        m = re.search(r"\{.*\}", texto, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return None
    return None


def limpar_query(q):
    """Rede de seguranca: remove vazamentos de numero de acordao da pergunta gerada.

    Mesmo pedindo no prompt, o LLM as vezes cita 'acordao 1234/2026'. Esses numeros
    'vazam' a identidade dos documentos e tornam a busca trivial (mascarando o ganho
    do enhancement). Aqui removemos os padroes NNNN/AAAA e expressoes 'com base no(s)
    acordao(s) ...' e normalizamos espacos/pontuacao.
    """
    q = re.sub(r"\d{1,5}\s*/\s*\d{4}", "", q)               # numeros tipo 1234/2026
    q = re.sub(r"(?i)\bn[ºo°]?\s*\d+\b", "", q)             # "nº 123"
    q = re.sub(r"(?i)\bac[oó]rd[aã]os?\b", "casos", q)      # acordao(s) -> casos
    q = re.sub(r"(?i)\bcom base (n|em)[^,.:;]*[,:]\s*", "", q)  # "com base nos casos ...,"
    q = re.sub(r"\s{2,}", " ", q)
    q = re.sub(r"\s+([,.;:?])", r"\1", q)
    q = re.sub(r"[,;:]\s*([,.;:?])", r"\1", q)
    q = q.strip(" ,;:-").strip()
    q = re.sub(r"^(e|que|mas|pois)\b[,\s]*", "", q, flags=re.I)  # tira conjuncao inicial solta
    q = q[:1].upper() + q[1:] if q else q
    return q.strip()


def gerar_benchmark(store, cliente, modelo, n, seed):
    """Modo padrao: n perguntas coloquiais de 1 documento (gabarito = aquele doc)."""
    indexados = store.filter_documents()
    if not indexados:
        raise RuntimeError("Indice vazio. Rode antes: python 01_indexar_opensearch.py")
    random.seed(seed)
    amostra = random.sample(indexados, min(n, len(indexados)))
    benchmark = []
    for i, d in enumerate(amostra, 1):
        docid = d.meta.get("id_original")
        par = extrair_json(_comum.gerar_texto(cliente, modelo, PROMPT_COLOQUIAL.format(texto=d.content[:2500]),
                                              max_tokens=700, temperature=0.4))
        if par and "pergunta" in par and docid:
            benchmark.append({"id": f"BQ{i:03d}", "query": str(par["pergunta"]).strip(),
                              "documentos_relevantes": [docid],
                              "resposta_esperada": str(par.get("resposta", "")).strip(),
                              "tipo": "coloquial"})
        print(f"  [{i}/{len(amostra)}] {benchmark[-1]['query'][:60] if benchmark else '(falhou)'}")
    with open(_comum.BENCHMARK, "w", encoding="utf-8") as f:
        json.dump(benchmark, f, ensure_ascii=False, indent=2)
    print(f"Benchmark gerado: {len(benchmark)} perguntas -> {_comum.BENCHMARK.name}")
    return benchmark


def gerar_benchmark_multidoc(store, cliente, modelo, n, seed, por_cluster=3, rel_por_cluster=4):
    """Modo --multi-doc: n perguntas TEMATICAS, cada uma com VARIOS docs relevantes.

    Agrupa os docs do indice por similaridade (KMeans) e, para cada cluster, define o
    GABARITO COESO = os 'rel_por_cluster' documentos MAIS PROXIMOS do centroide (tema
    coerente). A pergunta tematica e gerada a partir dos 'por_cluster' docs mais
    centrais. Gabarito coeso (em vez de 3 sementes soltas) evita recall artificialmente
    baixo e permite que Multi-Query/RAG-Fusion mostrem ganho ao "lancar uma rede maior".
    """
    indexados = store.filter_documents()
    if not indexados:
        raise RuntimeError("Indice vazio. Rode antes: python 01_indexar_opensearch.py")
    random.seed(seed)
    amostra = random.sample(indexados, min(60, len(indexados)))
    print(f"Embedando {len(amostra)} docs para clusterizar...")
    docs_emb = _comum.doc_embedder().run(documents=amostra)["documents"]
    vetores = np.array([d.embedding for d in docs_emb], dtype="float32")
    k = max(2, min(n, len(amostra) // por_cluster))
    km = KMeans(n_clusters=k, n_init=10, random_state=seed)
    rotulos = km.fit_predict(vetores)

    benchmark = []
    for c in range(k):
        idx = [j for j in range(len(amostra)) if rotulos[j] == c]
        if len(idx) < 2:
            continue
        # ordena os membros pela PROXIMIDADE ao centroide (mais coeso primeiro)
        centro = km.cluster_centers_[c]
        idx.sort(key=lambda j: float(np.linalg.norm(vetores[j] - centro)))
        relevantes_idx = idx[:rel_por_cluster]          # gabarito coeso (rede do tema)
        query_idx = idx[:min(por_cluster, len(idx))]    # docs centrais p/ gerar a pergunta
        trechos = "\n\n".join(amostra[j].content[:700] for j in query_idx)
        par = extrair_json(_comum.gerar_texto(cliente, modelo, PROMPT_TEMATICA.format(trechos=trechos),
                                              max_tokens=700, temperature=0.4))
        if par and "pergunta" in par:
            benchmark.append({"id": f"BT{c+1:03d}", "query": limpar_query(str(par["pergunta"])),
                              "documentos_relevantes": [amostra[j].meta.get("id_original") for j in relevantes_idx],
                              "resposta_esperada": str(par.get("resposta", "")).strip(),
                              "tipo": "tematica"})
        print(f"  cluster {c}: {len(idx)} docs (gabarito={len(relevantes_idx)}) -> "
              f"{benchmark[-1]['query'][:50] if benchmark else '(falhou)'}")
        if len(benchmark) >= n:
            break
    with open(BENCHMARK_MULTIDOC, "w", encoding="utf-8") as f:
        json.dump(benchmark, f, ensure_ascii=False, indent=2)
    print(f"Benchmark MULTI-DOC gerado: {len(benchmark)} perguntas tematicas -> {BENCHMARK_MULTIDOC.name}")
    return benchmark


def ids_de(docs):
    return [d.meta.get("id_original") for d in docs]


def busca_baseline(pipe, cliente, modelo, q, k, n):
    return ids_de(_comum.buscar(pipe, q)), 1, 0


def busca_multi(pipe, cliente, modelo, q, k, n):
    consultas = [q] + _comum.gerar_variacoes(cliente, modelo, q, n=n)
    listas = [_comum.buscar(pipe, c) for c in consultas]
    return ids_de(_comum.dedup_por_id(listas, k)), len(consultas), 1


def busca_stepback(pipe, cliente, modelo, q, k, n):
    geral = _comum.gerar_stepback(cliente, modelo, q)
    listas = [_comum.buscar(pipe, q), _comum.buscar(pipe, geral)]
    return ids_de(_comum.dedup_por_id(listas, k)), 2, 1


def busca_ragfusion(pipe, cliente, modelo, q, k, n):
    consultas = [q] + _comum.gerar_variacoes(cliente, modelo, q, n=n)
    listas = [_comum.buscar(pipe, c) for c in consultas]
    return ids_de(_comum.fundir_rrf(listas, k)), len(consultas), 1


def recall_hit(ids, relevantes, k):
    rel = set(relevantes)
    achados = set(ids[:k]) & rel
    recall = len(achados) / len(rel) if rel else 0.0
    return recall, (1.0 if achados else 0.0)


def main():
    parser = argparse.ArgumentParser(description="Benchmark Recall/Latencia/Custo (Aula 7).")
    parser.add_argument("--indice", default=_comum.INDICE_TCU)
    parser.add_argument("--n", type=int, default=15, help="tamanho do benchmark (se gerar)")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--variacoes", type=int, default=4)
    parser.add_argument("--multi-doc", action="store_true",
                        help="usa perguntas TEMATICAS com varios docs relevantes (clustering)")
    parser.add_argument("--rel-por-cluster", type=int, default=4,
                        help="tamanho do gabarito coeso por cluster (modo --multi-doc)")
    parser.add_argument("--gerar", action="store_true", help="forca regerar o benchmark do modo atual")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    _comum.carregar_env()
    cliente, modelo = _comum.groq_client()
    store = _comum.abrir_store(args.indice)
    pipe = _comum.montar_busca(store, args.top_k)
    k = args.top_k
    arquivo = BENCHMARK_MULTIDOC if args.multi_doc else _comum.BENCHMARK

    print("=" * 60)
    print(f"  BENCHMARK Query Enhancement - Aula 7  [modo: {'MULTI-DOC' if args.multi_doc else 'coloquial'}]")
    print("=" * 60)

    if args.gerar or not arquivo.exists():
        print("Gerando o benchmark...")
        if args.multi_doc:
            benchmark = gerar_benchmark_multidoc(store, cliente, modelo, args.n, args.seed,
                                                 rel_por_cluster=args.rel_por_cluster)
        else:
            benchmark = gerar_benchmark(store, cliente, modelo, args.n, args.seed)
    else:
        with open(arquivo, "r", encoding="utf-8") as f:
            benchmark = json.load(f)
        print(f"Benchmark existente: {len(benchmark)} perguntas (use --gerar para refazer).")

    estrategias = {
        "Baseline": busca_baseline,
        "Multi-Query": busca_multi,
        "Step-Back": busca_stepback,
        "RAG-Fusion": busca_ragfusion,
    }

    if not benchmark:
        print("\n[ERRO] O benchmark ficou VAZIO (o LLM nao gerou perguntas validas).")
        print("Causas comuns: modelo de RACIOCINIO (ex.: gpt-oss) devolvendo content vazio,")
        print("ou JSON malformado. Tente um modelo nao-reasoning no .env, por exemplo:")
        print("   GROQ_LLM_MODEL=llama-3.3-70b-versatile   (ou llama-3.1-8b-instant)")
        print("e rode novamente com --gerar.")
        return

    rel_medio = sum(len(q["documentos_relevantes"]) for q in benchmark) / max(1, len(benchmark))
    print(f"\nRodando {len(estrategias)} estrategias em {len(benchmark)} perguntas "
          f"(k={k}, ~{rel_medio:.1f} docs relevantes/pergunta)...")
    resultados = {}
    for nome, fn in estrategias.items():
        soma_recall = soma_hit = soma_lat = soma_buscas = soma_llm = 0.0
        for q in benchmark:
            t0 = time.perf_counter()
            try:
                ids, nb, nl = fn(pipe, cliente, modelo, q["query"], k, args.variacoes)
            except Exception as e:
                ids, nb, nl = [], 0, 0
                print(f"  ({nome} erro em {q['id']}: {str(e)[:60]})")
            dt = time.perf_counter() - t0
            rec, hit = recall_hit(ids, q["documentos_relevantes"], k)
            soma_recall += rec; soma_hit += hit; soma_lat += dt
            soma_buscas += nb; soma_llm += nl
        n = len(benchmark)
        resultados[nome] = {"Recall@k": soma_recall / n, "Hit@k": soma_hit / n,
                            "Latencia(s)": soma_lat / n, "Buscas/q": soma_buscas / n,
                            "LLM/q": soma_llm / n}
        print(f"  {nome} concluido.")

    print("\n" + "=" * 72)
    print(f"  RESULTADO (medias por estrategia)  [modo: {'MULTI-DOC' if args.multi_doc else 'coloquial'}]")
    print("=" * 72)
    cols = ["Hit@k", "Recall@k", "Latencia(s)", "Buscas/q", "LLM/q"]
    print(f"{'Estrategia':<13}" + "".join(f"{c:>13}" for c in cols))
    for nome, r in resultados.items():
        print(f"{nome:<13}" + "".join(f"{r[c]:>13.3f}" for c in cols))
    if args.multi_doc:
        print("\nNo modo MULTI-DOC (varios relevantes por pergunta), Multi-Query e RAG-Fusion "
              "tendem a SUPERAR o Baseline no Recall - e onde o enhancement compensa.")
    else:
        print("\nNo modo coloquial (1 relevante por pergunta), o Baseline costuma bastar; "
              "use --multi-doc para ver o enhancement ganhando.")


if __name__ == "__main__":
    main()
