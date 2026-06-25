# Índice — Aula 5: Avaliação e Observabilidade — RAGAS, DeepEval e LangFuse Avançado
## Medindo e Melhorando a Qualidade do Pipeline RAG Jurídico
### MBA em RAG & CAG Aplicados a Direito e Segurança Pública

**Aula:** 5 de 12 | **Carga:** 5h | **Proporção:** 30% teoria / 70% prática
**Pré-requisito:** Aulas 1–4 concluídas (Naive RAG, Advanced RAG, Hybrid Search, Contextual Retrieval)
**Stack:** RAGAS · DeepEval · LangFuse · Pandas · Matplotlib · **Groq** (LLM) · **Ollama** (embeddings BGE-M3) · **OpenSearch 3.x** (vector store)

---

## Estrutura de Arquivos

```
aula5/
│
├── INDICE_AULA5.md                                        ← Este arquivo
├── AVALIACAO_AULA5.md                                     ← Rubricas e critérios (professor)
│
├── teoria/
│   └── AULA5_TEORIA.md                                    ← Material teórico (8 seções)
│
├── labs/
│   ├── LAB1_Dataset_Avaliacao_GroundTruth.ipynb           ← Gera 50 pares Naive RAG → JSON/CSV
│   ├── LAB2_RAGAS_Baseline_Naive_RAG.ipynb                ← 4 métricas RAGAS no Naive RAG (#T01)
│   ├── LAB3_RAGAS_LangFuse_Scores_API.ipynb               ← RAGAS por par → LangFuse Scores API (+ @observe)
│   ├── LAB4_DeepEval_Testes_Unitarios.ipynb               ← 5 testes DeepEval (Faith/AnsRel/Hallu/Tox/Bias)
│   ├── LAB5_Dashboard_Naive_vs_Advanced.ipynb             ← Naive vs Advanced (Hybrid+Rerank) — gráficos
│   ├── LAB6_Analise_Erros_Faithfulness.ipynb              ← Diagnóstico + correção de Faith<0.70
│   └── LAB7_LangFuse_Avaliacao_Continua.ipynb             ← Datasets, Runs, Multi-Scores, Sessions, Annotation, Drift, Cron
│
├── exemplos/   (atualmente vazio — material em laboratórios)
│
└── datasets/
    └── corpus_avaliacao_aula5.json                        ← 50 pares QA jurídicos com ground-truth
```

---

## Stack Tecnológica (alinhada à Aula 3+)

| Componente | Ferramenta | Papel |
|---|---|---|
| LLM gerador & judge | **Groq Cloud** `llama-3.1-8b-instant` via `langchain_groq.ChatGroq` | Geração RAG e LLM judge para RAGAS/DeepEval |
| LLM fallback | **Ollama** `llama3.2:3b` via `langchain_ollama.ChatOllama` | Substituto automático quando Groq indisponível |
| Embeddings | **Ollama** `bge-m3` (dim=1024) via `langchain_ollama.OllamaEmbeddings` | Vetorização para kNN no OpenSearch |
| Vector Store | **OpenSearch 3.x** (índice `corpus_juridico_aula4` da Aula 4) | Recuperação kNN + BM25 (hybrid) |
| Framework de avaliação | **RAGAS** (≥0.1.16) + `LangchainLLMWrapper` / `LangchainEmbeddingsWrapper` | 4 métricas RAG |
| Testes unitários LLM | **DeepEval** (≥0.21) + wrapper custom `DeepEvalBaseLLM` | 5 métricas: Faith/AnsRel/Hallu/Tox/Bias |
| Observabilidade | **LangFuse** (Scores API + `@observe`) | Traces + scores por consulta |
| Manipulação/visualização | **Pandas / Matplotlib / Seaborn** | Dashboards e CSVs |

---

## Roteiro da Aula (5 horas)

| Bloco | Duração | Tipo | Conteúdo | Arquivo |
|---|---|---|---|---|
| **1. Por que avaliar RAG?** | 20 min | Teoria | Alucinação silenciosa, métricas clássicas vs. LLM-as-judge | `teoria/AULA5_TEORIA.md §1–2` |
| **2. Framework RAGAS** | 25 min | Teoria | 4 métricas, pipeline, metas mínimas | `teoria/AULA5_TEORIA.md §3–4` |
| **3. LAB 1 — Ground-Truth** | 40 min | Prática | Pipeline Naive RAG (Groq+Ollama+OpenSearch) gera dataset de 50 pares | `labs/LAB1_Dataset_Avaliacao_GroundTruth.ipynb` |
| **4. LAB 2 — RAGAS Baseline** | 45 min | Prática | 4 métricas RAGAS no Naive (LLM judge: Groq) | `labs/LAB2_RAGAS_Baseline_Naive_RAG.ipynb` |
| **5. LangFuse Scores API** | 15 min | Teoria | Scores API, `@observe`, quality gate | `teoria/AULA5_TEORIA.md §5` |
| **6. LAB 3 — RAGAS + LangFuse** | 30 min | Prática | Pipeline calcula RAGAS por par e envia ao LangFuse | `labs/LAB3_RAGAS_LangFuse_Scores_API.ipynb` |
| **7. DeepEval** | 15 min | Teoria | Testes unitários LLM, integração com pytest | `teoria/AULA5_TEORIA.md §6` |
| **8. LAB 4 — DeepEval** | 30 min | Prática | 5 testes (Faithfulness, AnswerRelevancy, Hallucination, Toxicity, Bias) | `labs/LAB4_DeepEval_Testes_Unitarios.ipynb` |
| **9. LAB 5 — Dashboard** | 30 min | Prática | Naive vs Advanced (Hybrid+Rerank) com Groq | `labs/LAB5_Dashboard_Naive_vs_Advanced.ipynb` |
| **10. LAB 6 — Análise de Erros** | 25 min | Prática | Diagnóstico + 3 correções para Faith<0.70 | `labs/LAB6_Analise_Erros_Faithfulness.ipynb` |
| **11. LAB 7 — Avaliação Contínua com LangFuse** | 30 min | Prática | Datasets versionados, Dataset Runs, Multi-Scores (NUMERIC/CATEG/BOOL), Sessions, Annotation Queue, Prompt Mgmt, Drift Detection, `@observe`+CallbackHandler, Cron diário | `labs/LAB7_LangFuse_Avaliacao_Continua.ipynb` |

---

## Variáveis `.env` (padrão do curso)

```
GROQ_API_KEY=gsk_...
GROQ_LLM_MODEL=llama-3.1-8b-instant
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_LLM_MODEL=llama3.2:3b
OLLAMA_EMBED_MODEL=bge-m3
OPENSEARCH_HOST=localhost
OPENSEARCH_PORT=9200
OPENSEARCH_USER=admin
OPENSEARCH_PASSWORD=admin
INDEX_NAME=corpus_juridico_aula4
LANGFUSE_HOST=http://localhost:3000
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
```

---

## Metas de Qualidade RAGAS (conforme syllabus)

| Métrica | Meta Mínima | Descrição |
|---|---|---|
| **Faithfulness** | ≥ 0.80 | Respostas fundamentadas nos contextos (sem alucinação) |
| **Answer Relevancy** | ≥ 0.75 | Resposta pertinente à pergunta |
| **Context Recall** | ≥ 0.70 | Ground-truth coberto pelos contextos recuperados |
| **Context Precision** | ≥ 0.70 | Contextos recuperados são realmente relevantes |

---

## Objetivos de Aprendizagem

1. Calcular as 4 métricas RAGAS usando **Groq como LLM judge** e **Ollama BGE-M3 como embedding judge** (via wrappers LangChain)
2. **Integrar avaliação RAGAS com LangFuse Scores API** — quality gate por consulta com `@observe`
3. Escrever **testes DeepEval** com judge customizado (`DeepEvalBaseLLM` envolvendo Groq/Ollama)
4. **Comparar pipelines** Naive vs Advanced (Hybrid+Rerank) com dashboards
5. **Diagnosticar falhas** de Faithfulness e aplicar correções automatizadas (retrieval/prompt/rewrite)
6. **Operar avaliação contínua** com LangFuse: datasets versionados, dataset runs, multi-scores (numeric/categorical/boolean), sessions/users, annotation queues, prompt management, drift detection e jobs agendados

---

## Avaliação

| Entregável | Peso | Lab |
|---|---|---|
| Dataset ground-truth com 50 pares exportado | 15% | LAB1 |
| 4 métricas RAGAS calculadas (≥1 acima da meta) | 25% | LAB2 |
| Pipeline RAGAS → LangFuse com `@observe` funcionando | 20% | LAB3 |
| 5 testes DeepEval executados (≥3 passando) | 20% | LAB4 |
| Dashboard Naive vs Advanced com gráficos | 12% | LAB5 |
| Análise de erros (≥3 queries diagnosticadas + corrigidas) | 5% | LAB6 |
| Avaliação contínua: dataset run + ≥3 capacidades LangFuse exploradas (Sessions/Annotation/Drift/Prompt Mgmt) | 8% | LAB7 |

---

## Referências (ABNT)

ES, S. et al. **RAGAS: Automated Evaluation of Retrieval Augmented Generation**. arXiv:2309.15217, 2023.

CONFIDENT AI. **DeepEval — The Open-Source LLM Evaluation Framework**. <https://docs.confident-ai.com>.

LANGFUSE. **Scores API + `@observe` decorator**. <https://langfuse.com/docs/scores>.

LANGFUSE. **Datasets and Experiments**. <https://langfuse.com/docs/datasets>.

LANGFUSE. **Sessions & Users**. <https://langfuse.com/docs/tracing-features/sessions>.

LANGFUSE. **Prompt Management**. <https://langfuse.com/docs/prompts/get-started>.

LANGFUSE. **Annotation queues**. <https://langfuse.com/docs/scores/annotation>.

GROQ INC. **Groq API Documentation**. <https://console.groq.com/docs>.

OLLAMA. **bge-m3 model card**. <https://ollama.com/library/bge-m3>.

OPENSEARCH PROJECT. **Hybrid / Vector Search**. 3.0 Docs. <https://docs.opensearch.org/3.0/vector-search/>.

RAGAS. **Customising LLMs and Embeddings — `LangchainLLMWrapper` / `LangchainEmbeddingsWrapper`**. <https://docs.ragas.io/en/stable/howtos/customisations/customize_models.html>.

BOGAN, R. et al. **Introducing reciprocal rank fusion for hybrid search**. OpenSearch Blog, 2025.

TRIBUNAL DE CONTAS DA UNIÃO. **Acórdãos 2026 — base completa** (usada como vector store da Aula 4).
