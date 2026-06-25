# Scripts da Aula 4 — Guia de Uso

Scripts Python simples, de linha de comando, que reproduzem os assuntos práticos
da Aula 4 (**OpenSearch completo: Dense, Hybrid Search, Neural Sparse e Contextual Retrieval**):

**Haystack** · **OpenSearch** (BM25 + kNN + rank_features) · **Ollama** (embeddings) · **SPLADE** (neural sparse) · **Groq** (LLM) · **LangFuse** (observabilidade)

---

## 1. Pré-requisitos

1. **Python 3.10+** com o ambiente virtual do curso ativado.
2. **OpenSearch** em `localhost:9200` — veja `GUIA_OPENSEARCH_WINDOWS.md` da **Aula 1**.
3. **Ollama** com o modelo de embedding: `ollama pull nomic-embed-text`.
4. **Chave da Groq** no `.env`.
5. **LangFuse** (opcional) — veja `GUIA_LANGFUSE_WINDOWS.md` da **Aula 3**.

As variáveis do `.env` são as mesmas das aulas anteriores (Groq, Ollama, OpenSearch, LangFuse).

> **Corpus:** os scripts usam `corpus_juridico_aula4_v2.json` (1100 acórdãos do TCU),
> que é o corpus alinhado com o gabarito das `queries_avaliacao_aula4.json`.

---

## 2. Instalação das dependências

Com o ambiente virtual **ativado**, dentro da pasta `scripts`:

```bash
pip install -r requirements.txt
```

> O modelo **SPLADE** (script 06) é baixado na 1ª execução (~500 MB–1 GB) e roda na CPU.

---

## 3. Ordem recomendada e como usar cada script

Rode tudo de dentro da pasta `aula4/scripts`.

### `00_check_ambiente.py` — confere se está tudo pronto
```bash
python 00_check_ambiente.py
python 00_check_ambiente.py --testar-groq
```

### `01_indexar_hibrido.py` — indexa o corpus (texto BM25 + vetor kNN)
Base para a busca híbrida. Precisa de OpenSearch + Ollama.
```bash
python 01_indexar_hibrido.py --recriar
python 01_indexar_hibrido.py --limite 200      # teste rápido
```

### `02_busca_hibrida_rrf.py` — BM25 x Densa x Híbrida (RRF)
Mostra as três listas lado a lado para a mesma pergunta.
```bash
python 02_busca_hibrida_rrf.py --query "operacao de credito com garantia da Uniao"
python 02_busca_hibrida_rrf.py --query "..." --top-k 5
```

### `03_contextual_retrieval.py` — Contextual Retrieval (#T09)
Enriquece cada documento com um contexto gerado pela Groq e indexa (índice próprio).
```bash
python 03_contextual_retrieval.py --recriar --limite 50
# depois compare:
python 02_busca_hibrida_rrf.py --indice aula4_contextual --query "..."
```

### `04_hybrid_rag.py` — RAG híbrido → Groq (com LangFuse)
Busca híbrida (BM25 + densa + RRF) e gera a resposta citando as fontes.
```bash
python 04_hybrid_rag.py --pergunta "Quais os requisitos para operacao de credito com garantia da Uniao?"
python 04_hybrid_rag.py --pergunta "..." --top-k 5
```

### `05_avaliar_busca.py` — métricas Densa vs Híbrida
MRR@k, Recall@k e NDCG@k usando as queries com gabarito.
```bash
python 05_avaliar_busca.py
python 05_avaliar_busca.py --top-k 10
```

### `06_neural_sparse.py` — Neural Sparse (SPLADE) + rank_features
Codifica em Python (SPLADE), indexa nos `rank_features` do OpenSearch e busca.
```bash
python 06_neural_sparse.py --recriar --limite 150
python 06_neural_sparse.py --so-buscar --query "operacao de credito com garantia da Uniao"
```

---

## 4. Resumo de dependências por script

| Script | Ollama | OpenSearch | SPLADE | Groq | LangFuse |
|--------|:------:|:----------:|:------:|:----:|:--------:|
| 00_check_ambiente | ✓ (checa) | ✓ (checa) | — | ✓ (checa) | ✓ (checa) |
| 01_indexar_hibrido | ✓ | ✓ | — | — | — |
| 02_busca_hibrida_rrf | ✓ | ✓ | — | — | — |
| 03_contextual_retrieval | ✓ | ✓ | — | ✓ | — |
| 04_hybrid_rag | ✓ | ✓ | — | ✓ | opcional |
| 05_avaliar_busca | ✓ | ✓ | — | — | — |
| 06_neural_sparse | — | ✓ | ✓ | — | — |

> `_comum.py` não é executado diretamente: é o arquivo auxiliar importado pelos outros scripts (carrega o `.env`, lê corpus/queries, prepara o LangFuse e as configurações).

---

## 5. Observações

- **Por que o corpus v2 (1100 docs)?** O gabarito das queries aponta para IDs de acórdãos que só existem no v2; por isso a avaliação (05) e o neural sparse (06) usam ele.
- **Neural Sparse sem ML Commons:** aqui o SPLADE roda em Python e os pesos vão para o campo `rank_features` do OpenSearch (a busca usa cláusulas `rank_feature`). É uma versão didática; em produção o OpenSearch pode hospedar o modelo via ML Commons (como no LAB4).
- **LangFuse:** ligado automaticamente no `04` quando há chaves no `.env`; sem elas, roda igual, só sem traces.
