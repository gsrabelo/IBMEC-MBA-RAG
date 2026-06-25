# Scripts da Aula 2 — Guia de Uso

Scripts Python simples, de linha de comando, que reproduzem os assuntos práticos
da Aula 2 (**Ingestão, Chunking e Naive RAG**) usando o stack do curso:

**Docling** (ingestão de PDF) · **LangChain text-splitters** (chunking) · **Haystack** (orquestração) · **OpenSearch** (busca vetorial kNN) · **Ollama** (embeddings) · **Groq** (LLM)

---

## 1. Pré-requisitos

1. **Python 3.10+** com o ambiente virtual do curso ativado.
2. **OpenSearch** rodando em `localhost:9200` — veja `GUIA_OPENSEARCH_WINDOWS.md` da **Aula 1**.
3. **Ollama** rodando com o modelo de embedding baixado: `ollama pull nomic-embed-text`.
4. **Chave da Groq** no `.env` (na raiz do projeto).

As variáveis do `.env` são as mesmas da Aula 1 (`GROQ_API_KEY`, `OLLAMA_BASE_URL`,
`OLLAMA_EMBED_MODEL`, `OPENSEARCH_HOST`, `OPENSEARCH_PORT`, `OPENSEARCH_USER`, `OPENSEARCH_PASS`).

> No lab, deixe `OPENSEARCH_USER` e `OPENSEARCH_PASS` vazios (single-node sem segurança).

---

## 2. Instalação das dependências

Com o ambiente virtual **ativado**, dentro da pasta `scripts`:

```bash
pip install -r requirements.txt
```

> Na **primeira** execução, o Docling baixa modelos (pode levar 1–2 min). Com `--ocr`,
> o EasyOCR também baixa modelos na primeira vez.

---

## 3. Ordem recomendada e como usar cada script

Rode tudo de dentro da pasta `aula2/scripts`.

### `00_check_ambiente.py` — confere se está tudo pronto
```bash
python 00_check_ambiente.py
python 00_check_ambiente.py --testar-groq
```

### `01_ingestao_docling.py` — extrai PDF com Docling (vs PyPDF2)
Converte um PDF em Markdown estruturado e compara com a extração simples do PyPDF2.
```bash
python 01_ingestao_docling.py
python 01_ingestao_docling.py --pdf ../datasets/Laudo-Minimal.pdf --ocr
python 01_ingestao_docling.py --salvar
```

### `02_chunking_comparar.py` — compara estratégias de chunking
Fixed, Recursive, Semantic (precisa do Ollama) e Header. Usa um acórdão de exemplo
por padrão; com `--pdf` extrai o texto de um PDF real.
```bash
python 02_chunking_comparar.py
python 02_chunking_comparar.py --estrategia recursive --chunk-size 800 --overlap 250
python 02_chunking_comparar.py --pdf ../datasets/Manual_DPCA_atualizado.pdf
```

### `03_indexar_chunks_opensearch.py` — indexa PDFs (precisa de OpenSearch + Ollama)
Extrai com Docling, quebra em chunks, gera embeddings e grava no índice kNN.
```bash
python 03_indexar_chunks_opensearch.py
python 03_indexar_chunks_opensearch.py --pdf ../datasets/Manual_DPCA_atualizado.pdf --recriar
python 03_indexar_chunks_opensearch.py --pasta ../datasets --estrategia recursive
```

### `04_naive_rag.py` — pergunta e resposta com RAG (precisa de tudo)
Busca os chunks no OpenSearch e gera a resposta com a Groq. Rode o `03` antes.
```bash
python 04_naive_rag.py --pergunta "Quais os requisitos da prisao preventiva?"
python 04_naive_rag.py --pergunta "O que o manual diz sobre oitiva?" --top-k 5
```

---

## 4. Resumo de dependências por script

| Script | Docling | Ollama | OpenSearch | Groq |
|--------|:-------:|:------:|:----------:|:----:|
| 00_check_ambiente | — | ✓ (checa) | ✓ (checa) | ✓ (checa) |
| 01_ingestao_docling | ✓ | — | — | — |
| 02_chunking_comparar | só com `--pdf` | só `semantic` | — | — |
| 03_indexar_chunks_opensearch | ✓ | ✓ | ✓ | — |
| 04_naive_rag | — | ✓ | ✓ | ✓ |

> `_comum.py` não é executado diretamente: é um arquivo auxiliar importado pelos outros scripts (carrega o `.env`, localiza os datasets e as configurações).
