# Scripts da Aula 6 — Guia de Uso

Scripts Python simples, de linha de comando, sobre **Indexação Avançada**:
**Parent-Child (#T07)**, **RAPTOR (#T08)** e **HyDE (#T05)**.

Stack (mesmo das aulas anteriores): **Haystack** · **OpenSearch** (vetores) ·
**Ollama** (embeddings) · **Groq** (LLM) · **LangFuse** (observabilidade no chat).


---

## 1. Dataset: por que e como gerar

O corpus original (`corpus_indexacao_avancada.json`, 10 docs curtos) serve para
**demonstrar** as técnicas, mas é pequeno e enviesado para **comparar** com números
confiáveis (docs curtos prejudicam Parent-Child; poucos docs/perguntas prejudicam RAPTOR).

Por isso o **`01_gerar_dataset.py`** monta um dataset melhor a partir dos **acórdãos do
TCU da Aula 4** (docs longos e numerosos — mesmo conteúdo do índice `aula4_hibrido`):

- amostra N documentos → salva `datasets/corpus_trabalho.json` (é o que as técnicas indexam);
- gera perguntas dos 3 tipos, cada uma marcada com `tecnica_ideal` + `documentos_relevantes` + `justificativa` → `datasets/perguntas_geradas.json`.

Quando esses arquivos existem, **todos os scripts passam a usá-los automaticamente**
(senão, caem no corpus original de 10 docs). Assim a comparação fica **eficiente e explicável**.

---

## 2. Pré-requisitos

1. **Python 3.10+** com o ambiente virtual do curso ativado.
2. **OpenSearch** em `localhost:9200` — veja `GUIA_OPENSEARCH_WINDOWS.md` da **Aula 1**.
3. **Ollama** com o modelo de embedding: `ollama pull nomic-embed-text`.
4. **Chave da Groq** no `.env`.
5. **LangFuse** (para o script 06) — veja `GUIA_LANGFUSE_WINDOWS.md` da **Aula 3**.
6. Para o `01`, é preciso ter o corpus da Aula 4 (`aula4/datasets/corpus_juridico_aula4_v2.json`).

```bash
pip install -r requirements.txt
```
> **RAGAS (script 05):** fixa `langchain-community==0.4.1` (a 0.4.2 quebra o import).

---

## 3. Ordem recomendada e como usar cada script

Rode tudo de dentro da pasta `aula6/scripts`.

### `00_check_ambiente.py` — confere se está tudo pronto
```bash
python 00_check_ambiente.py --testar-groq
```

### `01_gerar_dataset.py` — gera o dataset balanceado (TCU)
```bash
python 01_gerar_dataset.py --n-docs 80 --por-tecnica 6
```
Gera `corpus_trabalho.json` (amostra) e `perguntas_geradas.json` (18 perguntas: 6 por técnica).

### `02_parent_child.py` — Hierarchical / Parent-Child (#T07)
Arquitetura **toda no OpenSearch** (sem InMemory): um índice de **busca** (`<indice>`)
guarda só as **folhas com embedding** (kNN), e um índice de **árvore**
(`<indice>_arvore`) guarda **todos os nós (pais + folhas)** com os metadados de
hierarquia. O `AutoMergingRetriever` lê o `__parent_id` da folha recuperada e busca o
**pai por id** nesse índice da árvore. É **idempotente**: o split + embeddings rodam só
no `--recriar` (ou se algum índice estiver vazio); execuções normais só reabrem os
índices. Use `--inspecionar` para ver como a lib estrutura pais e filhos no OpenSearch.
```bash
python 02_parent_child.py --recriar
python 02_parent_child.py --pergunta "..."
python 02_parent_child.py --inspecionar    # mostra níveis, pais, filhos e parent_id
```

### `03_raptor.py` — RAPTOR (#T08)
```bash
python 03_raptor.py --recriar
python 03_raptor.py --pergunta "qual o panorama geral dos acordaos sobre ...?"
```

### `04_hyde.py` — Hypothetical Document Embeddings (#T05)
```bash
python 04_hyde.py --recriar
python 04_hyde.py --pergunta "..."
```

### `05_comparar_tecnicas.py` — comparação RAGAS (explicável)
Mostra médias por técnica **e** a matriz técnica × grupo-de-pergunta (`tecnica_ideal`):
espera-se a **diagonal mais alta** (cada técnica melhor no seu tipo de pergunta).
```bash
python 05_comparar_tecnicas.py
python 05_comparar_tecnicas.py --limite 9
```

### `06_chat_langfuse.py` — chat RAG com observabilidade + rastreabilidade
Consulta o índice do RAPTOR (`aula6_raptor`) e mostra, a cada resposta, a **origem**
de cada trecho: **RESUMO** de cluster (e de quais docs veio) ou **documento ORIGINAL**
(e seu cluster). Construa o índice do RAPTOR antes.
```bash
python 03_raptor.py --recriar       # cria o índice RAPTOR (com rastreabilidade)
python 06_chat_langfuse.py
```

> Fluxo típico: `01` (gera dataset) → `05` (compara). O `05` reconstrói os 3 índices
> a partir do `corpus_trabalho.json` e roda as perguntas geradas.

---

## 4. Resumo de dependências por script

| Script | OpenSearch | Ollama | Groq | RAGAS | LangFuse |
|--------|:----------:|:------:|:----:|:-----:|:--------:|
| 00_check_ambiente | ✓ (checa) | ✓ (checa) | ✓ (checa) | — | ✓ (checa) |
| 01_gerar_dataset | — | ✓ | ✓ | — | — |
| 02_parent_child | ✓ | ✓ | ✓ | — | — |
| 03_raptor | ✓ | ✓ | ✓ | — | — |
| 04_hyde | ✓ | ✓ | ✓ | — | — |
| 05_comparar_tecnicas | ✓ | ✓ | ✓ | ✓ | — |
| 06_chat_langfuse | ✓ | ✓ | ✓ | — | ✓ |

> `_comum.py` não é executado diretamente: carrega o `.env`, escolhe o corpus
> (trabalho > original), lê as perguntas (geradas > originais) e oferece os blocos
> Haystack (OpenSearch, embeddings Ollama, LLM Groq) + o helper `importar_script`.

---

## 5. Observações

- **Sem o `01`**, os scripts usam o corpus de 10 docs (modo demonstração).
- **Com o `01`**, usam a amostra do TCU (modo comparação imparcial).
- **RAPTOR** aqui é compacto (UMAP/KMeans + resumos via Groq → indexa originais + resumos).
- **LangFuse**: o `06` liga o tracing automaticamente quando há chaves no `.env`.
