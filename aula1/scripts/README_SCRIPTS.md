# Scripts da Aula 1 — Guia de Uso

Scripts Python simples, de linha de comando, que reproduzem os assuntos práticos
da Aula 1 usando o stack do curso:

**Haystack** (orquestração) · **OpenSearch** (busca vetorial kNN) · **Ollama** (embeddings) · **Groq** (LLM)

---

## 1. Pré-requisitos

Antes de rodar os scripts você precisa de:

1. **Python 3.10+** com o ambiente virtual do curso ativado.
2. **OpenSearch** rodando em `localhost:9200` — veja `GUIA_OPENSEARCH_WINDOWS.md`.
3. **Ollama** rodando com o modelo de embedding baixado:
   ```
   ollama pull nomic-embed-text
   ```
4. **Chave da Groq** configurada no arquivo `.env` (na raiz do projeto).

### Variáveis esperadas no `.env` (raiz do projeto)

```
GROQ_API_KEY=...sua_chave...
GROQ_LLM_MODEL=llama-3.1-8b-instant
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_EMBED_MODEL=nomic-embed-text
OPENSEARCH_HOST=localhost
OPENSEARCH_PORT=9200
OPENSEARCH_USER=
OPENSEARCH_PASS=
```

> No lab usamos OpenSearch single-node **sem HTTPS e sem senha** (security desabilitada).
> Deixe `OPENSEARCH_USER` e `OPENSEARCH_PASS` vazios nesse caso.

---

## 2. Instalação das dependências

Com o ambiente virtual **ativado**, dentro da pasta `scripts`:

```bash
pip install -r requirements.txt
```

---

## 3. Ordem recomendada e como usar cada script

Rode tudo de dentro da pasta `aula1/scripts`.

### `00_check_ambiente.py` — confere se está tudo pronto
Verifica Python, bibliotecas, `.env`, OpenSearch (+kNN), Ollama e Groq.
```bash
python 00_check_ambiente.py
python 00_check_ambiente.py --testar-groq     # faz 1 chamada real na Groq
```

### `01_nlp_basico_ptbr.py` — fundamentos de NLP (offline)
Tokenização, stemming e Bag-of-Words em português. A tokenização usa, por padrão,
um documento do corpus (`datasets/corpus_juridico_aula1.json`). Não precisa de serviços.
```bash
python 01_nlp_basico_ptbr.py
python 01_nlp_basico_ptbr.py --doc 4
python 01_nlp_basico_ptbr.py --texto "O juiz absolveu o reu por falta de provas."
```

### `02_embeddings_similaridade.py` — embeddings e similaridade (precisa do Ollama)
Gera os vetores dos documentos do corpus (`datasets/corpus_juridico_aula1.json`) e mostra
quais são parecidos. Com `--umap` salva um gráfico (`umap_documentos.png`).
```bash
python 02_embeddings_similaridade.py
python 02_embeddings_similaridade.py --n 10
python 02_embeddings_similaridade.py --umap
python 02_embeddings_similaridade.py --frase1 "prisao preventiva" --frase2 "soltura do reu"
```

### `03_metricas_retrieval.py` — métricas de qualidade de busca (precisa do Ollama)
Calcula Hit Rate@K, Recall@K, MRR e NDCG@K usando um gabarito.
```bash
python 03_metricas_retrieval.py
python 03_metricas_retrieval.py --k 1 3 5
```

### `04_indexar_opensearch.py` — indexa o corpus (precisa de OpenSearch + Ollama)
Gera os embeddings dos 10 documentos e grava no índice kNN do OpenSearch.
```bash
python 04_indexar_opensearch.py
python 04_indexar_opensearch.py --recriar      # apaga e indexa do zero
```

### `05_rag_minimo.py` — pergunta e resposta com RAG (precisa de tudo)
Busca no OpenSearch e gera a resposta com a Groq. Rode o `04` antes.
```bash
python 05_rag_minimo.py --pergunta "Quais os requisitos da prisao preventiva?"
python 05_rag_minimo.py --pergunta "O que e peculato?" --top-k 5
```

---

## 4. Resumo de dependências por script

| Script | Ollama | OpenSearch | Groq |
|--------|:------:|:----------:|:----:|
| 00_check_ambiente | ✓ (checa) | ✓ (checa) | ✓ (checa) |
| 01_nlp_basico_ptbr | — | — | — |
| 02_embeddings_similaridade | ✓ | — | — |
| 03_metricas_retrieval | ✓ | — | — |
| 04_indexar_opensearch | ✓ | ✓ | — |
| 05_rag_minimo | ✓ | ✓ | ✓ |

> `_comum.py` não é executado diretamente: é um arquivo auxiliar importado pelos outros scripts (carrega o `.env`, lê o corpus e as configurações).
