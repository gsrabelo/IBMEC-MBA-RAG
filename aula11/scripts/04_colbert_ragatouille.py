"""
04_colbert_pylate.py - ColBERT (late interaction) via PyLate vs busca densa.

ColBERT guarda um vetor por TOKEN e pontua por MaxSim (precisao alta). Usamos o PyLate
(LightOn), que roda ColBERT em PyTorch PURO - sem compilar extensao C++ (logo, funciona
no Windows/CPU SEM Visual Studio Build Tools, ao contrario do ragatouille/colbert classico).

Como o corpus e pequeno (30 docs), usamos `pylate.rank.rerank`: encodamos todos os
documentos e a query e ranqueamos por MaxSim - sem precisar montar indice PLAID. Depois
comparamos com a busca DENSA (bi-encoder, Ollama+OpenSearch) para a mesma query.

Precisa: pip install pylate   (puxa torch + sentence-transformers; baixa o ColBERT)
e o indice denso: python 01_indexar.py

Modelo: colbert-ir/colbertv2.0 (padrao). Para um ColBERT moderno (melhor em CPU), use
AULA11_COLBERT_MODEL=lightonai/GTE-ModernColBERT-v1 no .env.

Uso:
    python 04_colbert_ragatouille.py --pergunta "prazo de recurso de apelacao"
    python 04_colbert_ragatouille.py --pergunta "..." --top-k 5
"""

import argparse
import os

import _comum

_comum.carregar_env()


def resultados_colbert(pergunta, top_k):
    """ColBERT via PyLate (rank.rerank, sem indice/compilador). Devolve [(id, texto, score)]."""
    from pylate import models, rank

    docs = _comum.carregar_corpus()
    textos = [d["texto"] for d in docs]
    ids = [d["id"] for d in docs]
    por_id = {d["id"]: d["texto"] for d in docs}

    modelo_nome = os.getenv("AULA11_COLBERT_MODEL", "colbert-ir/colbertv2.0")
    print(f"Carregando ColBERT ({modelo_nome}) via PyLate... (1a vez baixa o modelo)")
    model = models.ColBERT(model_name_or_path=modelo_nome)

    docs_emb = model.encode(textos, is_query=False, show_progress_bar=False)
    q_emb = model.encode([pergunta], is_query=True, show_progress_bar=False)
    reranked = rank.rerank(documents_ids=[ids], queries_embeddings=q_emb,
                           documents_embeddings=[docs_emb])
    out = []
    for r in reranked[0][:top_k]:
        out.append((r["id"], por_id.get(r["id"], ""), r["score"]))
    return out


def main():
    parser = argparse.ArgumentParser(description="ColBERT (PyLate) vs busca densa.")
    parser.add_argument("--pergunta", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    print("=" * 60)
    print("  ColBERT (PyLate, late interaction) vs BUSCA DENSA - Aula 11")
    print("=" * 60)
    print(f"Pergunta: {args.pergunta}\n")

    # 1) busca densa (bi-encoder) - reusa o indice da aula
    store = _comum.abrir_store()
    if store.count_documents() == 0:
        print("[ATENCAO] indice denso vazio. Rode antes: python 01_indexar.py")
        return
    densos = _comum.buscar(_comum.montar_busca(store, args.top_k), args.pergunta)
    print("DENSA (bi-encoder):")
    for i, d in enumerate(densos, 1):
        print(f"  {i}. {d.score:.3f}  {d.meta.get('id_original')}  {d.content[:60]}")

    # 2) ColBERT (late interaction) via PyLate
    try:
        res = resultados_colbert(args.pergunta, args.top_k)
    except ImportError:
        print("\n[ERRO] instale: pip install pylate")
        return
    print("\nColBERT (late interaction / MaxSim):")
    for i, (cid, texto, score) in enumerate(res, 1):
        print(f"  {i}. {score:.3f}  {cid}  {texto[:60]}")

    print("\nLeitura: o ColBERT tende a casar melhor termos especificos (granularidade por "
          "token). Compare a ordem/precisao das duas listas. PyLate roda em CPU/Windows "
          "sem compilador (ao contrario do ragatouille/colbert classico).")


if __name__ == "__main__":
    main()
