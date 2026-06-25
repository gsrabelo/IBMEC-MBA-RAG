"""
06_neural_sparse.py - Neural Sparse Search (SPLADE) em Python + OpenSearch rank_features.

A busca esparsa neural (SPLADE) transforma um texto em um conjunto de
TOKENS COM PESO (ex.: {"credito": 2.1, "uniao": 1.4, ...}). Diferente do BM25,
ela faz "expansao": inclui termos relacionados que nem aparecem no texto. E
diferente do vetor denso: cada dimensao e uma palavra (interpretavel).

Como nao usamos o ML Commons do OpenSearch, fazemos o encoding AQUI no Python
com um modelo SPLADE e gravamos os pesos no campo `rank_features` do OpenSearch.
A busca usa varias clausulas `rank_feature` (uma por token da pergunta).

Precisa de OpenSearch. Na 1a vez baixa o modelo SPLADE (~500 MB-1 GB).
Encoding roda na CPU: use --limite para indexar um subconjunto.

Uso:
    python 06_neural_sparse.py --recriar --limite 150
    python 06_neural_sparse.py --query "operacao de credito com garantia da Uniao"
"""

import argparse
import re

import _comum

MODELO_SPLADE = "opensearch-project/opensearch-neural-sparse-encoding-multilingual-v1"
INDICE_PADRAO = "aula4_neural_sparse"
_RE_TOKEN_OK = re.compile(r"[0-9A-Za-zÀ-ÿ_]+")


def limpar_token(tok):
    """Remove marcadores de subpalavra e mantem so tokens 'limpos' (sem ponto etc.)."""
    tok = tok.replace("▁", "").replace("##", "").strip()  # ▁ = '_' do sentencepiece
    return tok if (tok and _RE_TOKEN_OK.fullmatch(tok)) else ""


def carregar_modelo():
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    print(f"Carregando SPLADE {MODELO_SPLADE} (1a vez baixa o modelo)...")
    tok = AutoTokenizer.from_pretrained(MODELO_SPLADE)
    model = AutoModelForMaskedLM.from_pretrained(MODELO_SPLADE)
    model.eval()
    return tok, model


def encode_sparse(texto, tokenizer, model, max_tokens=200):
    """Converte um texto em {token: peso} no estilo SPLADE (max-pooling)."""
    import torch

    inputs = tokenizer(texto, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        logits = model(**inputs).logits                      # (1, seq, vocab)
    mask = inputs["attention_mask"].unsqueeze(-1)            # (1, seq, 1)
    vec = (torch.log1p(torch.relu(logits)) * mask).max(dim=1).values.squeeze(0)  # (vocab,)

    pesos = {}
    for idx in torch.nonzero(vec).squeeze(-1).tolist():
        peso = float(vec[idx])
        if peso <= 0:
            continue
        token = limpar_token(tokenizer.convert_ids_to_tokens(idx))
        if token:
            pesos[token] = max(pesos.get(token, 0.0), round(peso, 4))
    # mantem os tokens de maior peso
    return dict(sorted(pesos.items(), key=lambda kv: -kv[1])[:max_tokens])


def cliente_opensearch():
    import os

    from opensearchpy import OpenSearch

    cfg = _comum.config_opensearch()
    host = os.getenv("OPENSEARCH_HOST", "localhost")
    porta = int(os.getenv("OPENSEARCH_PORT", "9200"))
    auth = (cfg["usuario"], cfg["senha"]) if cfg["usuario"] else None
    return OpenSearch(hosts=[{"host": host, "port": porta}], http_auth=auth,
                      use_ssl=False, verify_certs=False, ssl_show_warn=False)


def criar_indice(client, indice, recriar):
    if recriar and client.indices.exists(index=indice):
        client.indices.delete(index=indice)
    if not client.indices.exists(index=indice):
        client.indices.create(index=indice, body={
            "mappings": {"properties": {
                "id_original": {"type": "keyword"},
                "titulo": {"type": "text"},
                "sparse_embedding": {"type": "rank_features"},
            }},
        })
        print(f"Indice '{indice}' criado.")


def indexar(client, indice, tokenizer, model, limite):
    docs = _comum.carregar_corpus(limite=limite)
    print(f"Codificando e indexando {len(docs)} documentos (SPLADE na CPU)...")
    for n, d in enumerate(docs, 1):
        sparse = encode_sparse(d["texto"], tokenizer, model)
        client.index(index=indice, id=d["id"], body={
            "id_original": d["id"], "titulo": d.get("titulo", ""),
            "sparse_embedding": sparse,
        })
        if n % 25 == 0:
            print(f"  ...{n} documentos")
    client.indices.refresh(index=indice)
    print(f"Total indexado: {client.count(index=indice)['count']}")


def buscar(client, indice, tokenizer, model, query, k):
    qvec = encode_sparse(query, tokenizer, model, max_tokens=50)
    print("\nTokens da pergunta (top-10, interpretabilidade SPLADE):")
    for t, w in list(qvec.items())[:10]:
        print(f"  {t:20} {w:.3f}")
    should = [{"rank_feature": {"field": f"sparse_embedding.{t}", "boost": w}} for t, w in qvec.items()]
    body = {"size": k, "query": {"bool": {"should": should}}, "_source": ["titulo", "id_original"]}
    hits = client.search(index=indice, body=body)["hits"]["hits"]
    print(f"\nTOP-{k} (neural sparse):")
    print("-" * 50)
    for i, h in enumerate(hits, 1):
        print(f"  #{i} [{h['_score']:.3f}] {h['_source'].get('titulo', '')[:65]}")


def main():
    parser = argparse.ArgumentParser(description="Neural Sparse (SPLADE) + rank_features.")
    parser.add_argument("--indice", default=INDICE_PADRAO, help="indice no OpenSearch")
    parser.add_argument("--query", default="operacao de credito com garantia da Uniao",
                        help="pergunta de busca")
    parser.add_argument("--limite", type=int, default=150, help="quantos docs indexar (0 = todos)")
    parser.add_argument("--recriar", action="store_true", help="recria o indice e reindexa")
    parser.add_argument("--so-buscar", action="store_true", help="nao indexa, so faz a busca")
    args = parser.parse_args()

    _comum.carregar_env()
    print("=" * 60)
    print("  NEURAL SPARSE SEARCH (SPLADE) - Aula 4")
    print("=" * 60)

    client = cliente_opensearch()
    tokenizer, model = carregar_modelo()

    if not args.so_buscar:
        criar_indice(client, args.indice, args.recriar)
        indexar(client, args.indice, tokenizer, model, args.limite)

    print(f"\nQuery: {args.query}")
    buscar(client, args.indice, tokenizer, model, args.query, k=5)
    print("\nDica: repare que o SPLADE inclui termos relacionados (expansao) que nem "
          "estao no texto original - isso ajuda quando a pergunta usa outras palavras.")


if __name__ == "__main__":
    main()
