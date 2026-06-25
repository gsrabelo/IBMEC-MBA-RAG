# Scripts da Aula 3 — Guia de Uso

Scripts Python simples, de linha de comando, que reproduzem os assuntos práticos
da Aula 3 (**Advanced RAG e Modular RAG**) usando o stack do curso:

**Haystack** (orquestração) · **OpenSearch** (kNN) · **Ollama** (embeddings) · **BGE-Reranker** (reranking) · **Groq** (LLM) · **LangFuse** (observabilidade)

---

## 1. Pré-requisitos

1. **Python 3.10+** com o ambiente virtual do curso ativado.
2. **OpenSearch** em `localhost:9200` — veja `GUIA_OPENSEARCH_WINDOWS.md` da **Aula 1**.
3. **Ollama** com o modelo de embedding: `ollama pull nomic-embed-text`.
4. **Chave da Groq** no `.env`.
5. **LangFuse** (opcional, mas recomendado) — veja `GUIA_LANGFUSE_WINDOWS.md`.

Variáveis do `.env` (raiz do projeto): as mesmas das aulas anteriores, mais o LangFuse:

```
GROQ_API_KEY=...            GROQ_LLM_MODEL=llama-3.1-8b-instant
OLLAMA_BASE_URL=http://localhost:11434   OLLAMA_EMBED_MODEL=nomic-embed-text
OPENSEARCH_HOST=localhost   OPENSEARCH_PORT=9200   OPENSEARCH_USER=   OPENSEARCH_PASS=
LANGFUSE_PUBLIC_KEY=pk-lf-...   LANGFUSE_SECRET_KEY=sk-lf-...   LANGFUSE_BASE_URL=http://localhost:3000
```

---

## 2. Instalação das dependências

Com o ambiente virtual **ativado**, dentro da pasta `scripts`:

```bash
pip install -r requirements.txt
```

> O reranker `BAAI/bge-reranker-v2-m3` é baixado na **primeira** execução (~560 MB).
>
> O reranking usa o `TransformersSimilarityRanker` do Haystack (estável com a
> `sentence-transformers 3.x` do venv). Ele exibe um aviso de *"legacy"* — é
> **inofensivo**, o script funciona normalmente. Veja a seção 5 (Problemas comuns).

---

## 3. Ordem recomendada e como usar cada script

Rode tudo de dentro da pasta `aula3/scripts`.

### `00_check_ambiente.py` — confere se está tudo pronto (inclui LangFuse)
```bash
python 00_check_ambiente.py
python 00_check_ambiente.py --testar-groq
```

### `01_query_rewriting.py` — reescrita de query (precisa da Groq)
Paraphrase, HyDE-Lite e Step-Back. Não precisa de OpenSearch nem Ollama.
```bash
python 01_query_rewriting.py
python 01_query_rewriting.py --query "Podem prender alguem sem mandado?"
python 01_query_rewriting.py --tecnica hyde
```

### `02_indexar_opensearch.py` — indexa o corpus (precisa de OpenSearch + Ollama)
Indexa os ~80 documentos do `corpus_juridico_aula3.json` no índice kNN.
```bash
python 02_indexar_opensearch.py
python 02_indexar_opensearch.py --recriar
```

### `03_reranking_bge.py` — reranking com BGE (precisa de OpenSearch + Ollama)
Busca top-N e reordena com o cross-encoder, mostrando o antes/depois.
```bash
python 03_reranking_bge.py --query "O suspeito pode ficar calado no interrogatorio?"
python 03_reranking_bge.py --top-n 10 --top-k 5
```

### `04_advanced_rag.py` — Advanced RAG completo (precisa de tudo)
Reescrita (opcional) → busca → rerank → resposta na Groq, com trace no LangFuse.
```bash
python 04_advanced_rag.py --pergunta "Podem prender alguem sem mandado?"
python 04_advanced_rag.py --pergunta "Quais os requisitos da prisao preventiva?" --rewrite hyde --top-n 10 --rerank-k 4
```

> Use sempre uma **pergunta real** em `--pergunta`. Com `--rewrite`, a pergunta é
> reescrita antes da busca; um texto vazio ou `"..."` faz o rewrite "não entender" a pergunta.

---

## 4. Resumo de dependências por script

| Script | Ollama | OpenSearch | BGE-Reranker | Groq | LangFuse |
|--------|:------:|:----------:|:------------:|:----:|:--------:|
| 00_check_ambiente | ✓ (checa) | ✓ (checa) | — | ✓ (checa) | ✓ (checa) |
| 01_query_rewriting | — | — | — | ✓ | — |
| 02_indexar_opensearch | ✓ | ✓ | — | — | — |
| 03_reranking_bge | ✓ | ✓ | ✓ | — | — |
| 04_advanced_rag | ✓ | ✓ | ✓ | ✓ | opcional |

> `_comum.py` não é executado diretamente: é um arquivo auxiliar importado pelos outros scripts (carrega o `.env`, lê o corpus, prepara o LangFuse e as configurações).
>
> **LangFuse** é ligado automaticamente quando as chaves estão no `.env` (auto-instrumentação do Haystack). Sem as chaves, o `04` roda normalmente, apenas sem enviar traces.

---

## 5. Problemas comuns

| Mensagem / sintoma | Causa | Solução |
|---|---|---|
| `TransformersSimilarityRanker is considered legacy` | Aviso de depreciação do Haystack | **Inofensivo** — pode ignorar; o reranking funciona. |
| `CrossEncoder.__init__() got an unexpected keyword argument 'model_name_or_path'` | O ranker novo exige `sentence-transformers >= 4.0`; o venv tem 3.x | Os scripts já usam o `TransformersSimilarityRanker` (estável no 3.x). Se quiser o ranker novo, atualize: `pip install -U "sentence-transformers>=4"` (teste antes as Aulas 1 e 2). |
| `Missing mandatory input 'query' for component 'ranker'` | Versão antiga do `04` | Já corrigido — o `pipe.run` passa `query` ao ranker. Atualize o arquivo. |
| Rewrite responde "não foi possível identificar a pergunta" | `--pergunta` vazio ou `"..."` | Passe uma pergunta real. |
| `LangFuse: desligado` | Chaves ausentes no `.env` | Suba o LangFuse (`GUIA_LANGFUSE_WINDOWS.md`) e preencha as chaves; o `04` roda igual sem elas, só sem traces. |
| OpenSearch/Ollama não respondem | Serviços fora do ar | Veja `GUIA_OPENSEARCH_WINDOWS.md` (Aula 1) e rode `ollama serve`. Confira com `python 00_check_ambiente.py`. |
