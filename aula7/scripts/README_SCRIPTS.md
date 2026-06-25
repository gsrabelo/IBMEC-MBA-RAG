# Scripts da Aula 7 — Guia de Uso

Scripts Python simples, de linha de comando, sobre **Query Enhancement**:
**Multi-Query RAG (#T10)**, **Step-Back Prompting (#T11)** e **RAG-Fusion (#T12)**.

Stack: **Haystack** · **OpenSearch** (vetores) · **Ollama** (embeddings) · **Groq** (LLM) · **LangFuse** (observabilidade).

> **Corpus/índice:** a Aula 7 **reaproveita os acórdãos do TCU** (o índice `aula4_hibrido`
> da Aula 4). As 3 técnicas mudam só a **estratégia de query** — o índice é o mesmo.

---

## 1. Como funciona o conjunto

As três técnicas atacam o mesmo problema (a pergunta usa palavras diferentes dos documentos):

- **Multi-Query** — o LLM gera N reformulações; busca por todas; junta (dedup).
- **Step-Back** — o LLM cria uma pergunta mais geral; busca a específica + a geral.
- **RAG-Fusion** — gera N reformulações; busca por todas; funde os rankings com **RRF**.

O `05_benchmark` mede o **trade-off**: quanto cada técnica melhora o **Recall** vs.
quanto custa a mais (mais buscas + 1 chamada de LLM) e a **latência**.

---

## 2. Pré-requisitos

1. **Python 3.10+** com o ambiente virtual do curso ativado.
2. **OpenSearch** em `localhost:9200` com o índice do TCU. Use o `01` (reaproveita
   `aula4_hibrido`; se vazio, indexa o corpus do TCU).
3. **Ollama** com o modelo de embedding: `ollama pull nomic-embed-text`.
4. **Chave da Groq** no `.env`.
5. **LangFuse** (para o `06`) — veja `GUIA_LANGFUSE_WINDOWS.md` da **Aula 3**.

```bash
pip install -r requirements.txt
```

---

## 3. Ordem recomendada e como usar cada script

Rode tudo de dentro da pasta `aula7/scripts`.

### `00_check_ambiente.py` — confere se está tudo pronto
```bash
python 00_check_ambiente.py --testar-groq
```

### `01_indexar_opensearch.py` — garante/reusa o índice do TCU
```bash
python 01_indexar_opensearch.py            # reaproveita aula4_hibrido se existir
python 01_indexar_opensearch.py --recriar  # reindexa o corpus do TCU
```

### `02_multi_query.py` — Multi-Query RAG (#T10)
Dois modos para comparar: **manual** (padrão — busca cada variação e deduplica com
`_comum.dedup_por_id`) e **`--nativo`** (usa o componente oficial
`MultiQueryEmbeddingRetriever` do Haystack, com retrieval **paralelo** e dedup interno
`_deduplicate_documents` — mesma regra: por id, maior score). Em ambos as variações
vêm do LLM (Groq); o componente nativo não gera variações, só recebe a lista pronta.
```bash
python 02_multi_query.py --pergunta "o gestor pode ser multado pelo tribunal de contas?" --n 4
python 02_multi_query.py --nativo   # MultiQueryEmbeddingRetriever (retrieval paralelo)
```

### `03_step_back.py` — Step-Back Prompting (#T11)
```bash
python 03_step_back.py --pergunta "Qual foi a decisão do TCU sobre fiscalização sobre armas de fogo?"
```

### `04_rag_fusion.py` — RAG-Fusion (#T12)
```bash
python 04_rag_fusion.py --pergunta "quando as contas sao julgadas irregulares?" --n 4
```

### `05_benchmark.py` — Recall vs Latência vs Custo (todas as técnicas)
Gera o benchmark a partir do TCU (na 1ª vez) e compara **Baseline, Multi-Query,
Step-Back e RAG-Fusion** por Hit@k, Recall@k, latência e custo.

Dois modos:
- **padrão** (1 doc relevante por pergunta, "lookup"): o Baseline costuma bastar.
- **`--multi-doc`** (perguntas temáticas com vários docs relevantes, via clustering):
  é onde **Multi-Query/RAG-Fusion ganham** no Recall. Use `--top-k 10`.

No modo `--multi-doc`, o **gabarito é coeso**: os `--rel-por-cluster` (padrão 4) documentos
mais próximos do centroide de cada cluster (tema coerente). Isso evita o recall
artificialmente baixo de quando o gabarito eram só 3 sementes soltas com query muito
abstrata. **Importante:** se você já tem um `benchmark_multidoc.json` antigo, rode com
`--gerar` para refazê-lo no novo formato.

```bash
python 05_benchmark.py                              # modo coloquial (1 relevante/pergunta)
python 05_benchmark.py --multi-doc --top-k 10 --gerar   # refaz o benchmark temático coeso
python 05_benchmark.py --multi-doc --top-k 10 --rel-por-cluster 5
```

> Nota: o **Multi-Query** usa `dedup` por melhor score (não RRF). Como o score de cosseno
> não é comparável entre queries diferentes, uma variação pode empurrar um relevante para
> fora do top-k — por isso o Multi-Query pode ficar **abaixo** do Baseline em gabaritos
> fracos. O **RAG-Fusion** usa RRF (baseado em rank) e é mais robusto a isso. Mantivemos
> a distinção de propósito (é a lição: por que RRF > dedup).

### `06_chat_langfuse.py` — testa as técnicas com observabilidade
Cada técnica é um **pipeline Haystack completo**, incluindo a **reescrita da query**
(variações/step-back) feita por componentes `OpenAIGenerator` — então a
auto-instrumentação do LangFuse **captura também essas chamadas de LLM** no mesmo
trace (não só a geração da resposta). Componentes customizados em `_componentes.py`
(`MontarConsultas`, `BuscarMultiplas`). Cada técnica gera um trace nomeado
`chat-aula7-<tecnica>` para comparar lado a lado.
```bash
python 06_chat_langfuse.py                 # roda TODAS as técnicas por pergunta
python 06_chat_langfuse.py --tecnica rag_fusion
```

---

## 4. Resumo de dependências por script

| Script | OpenSearch | Ollama | Groq | LangFuse |
|--------|:----------:|:------:|:----:|:--------:|
| 00_check_ambiente | ✓ (checa) | ✓ (checa) | ✓ (checa) | ✓ (checa) |
| 01_indexar_opensearch | ✓ | ✓ | — | — |
| 02_multi_query | ✓ | ✓ | ✓ | — |
| 03_step_back | ✓ | ✓ | ✓ | — |
| 04_rag_fusion | ✓ | ✓ | ✓ | — |
| 05_benchmark | ✓ | ✓ | ✓ | — |
| 06_chat_langfuse | ✓ | ✓ | ✓ | ✓ |

> `_comum.py` não é executado diretamente: carrega o `.env`, lê o corpus do TCU,
> e oferece os blocos (busca densa, geração de variações/step-back, dedup e RRF).

---

## 5. Observações

- **Reaproveitamento do TCU:** o `01` não reindexa se o `aula4_hibrido` já tiver documentos.
  Garanta que esse índice contém o corpus completo (rode a Aula 4 ou `01 --recriar`).
- **Benchmark gerado:** o `05` cria perguntas **coloquiais** a partir dos acórdãos
  (gabarito = o documento de origem), ideais para exercitar o query enhancement.
- **Multi-Query vs RAG-Fusion:** ambos geram variações; o RAG-Fusion **funde por RRF**
  (mais robusto), o Multi-Query apenas deduplica.
- **LangFuse:** o `06` liga o tracing automaticamente quando há chaves no `.env`.
