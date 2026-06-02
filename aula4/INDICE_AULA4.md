# Índice — Aula 4: OpenSearch Completo — Dense, Hybrid Search, Neural Sparse e Contextual Retrieval
## Ollama BGE-M3 (connector ML Commons) · SPLADE Multilíngue Pré-Treinado · Groq LLM · RRF · Search Pipelines
### MBA em RAG & CAG Aplicados a Direito e Segurança Pública

**Aula:** 4 de 12 | **Carga:** 5h | **Proporção:** 30% teoria / 70% prática
**Pré-requisito:** Aula 3 concluída (Advanced RAG + Modular RAG + LangFuse) | **Stack:** OpenSearch 3.x · Neural Search Plugin · ML Commons Plugin · Ollama BGE-M3 (connector) · SPLADE Multilíngue (pré-treinado) · RRF · Search Pipelines · Groq + Ollama LLM · RAGAS · LangFuse

---

## Estrutura de Arquivos

```
aula4/
│
├── INDICE_AULA4.md                                      ← Este arquivo
├── AVALIACAO_AULA4.md                                   ← Rubricas e critérios (professor)
│
├── teoria/
│   └── AULA4_TEORIA.md                                  ← Material teórico completo (7 seções)
│
├── labs/
│   ├── LAB1_OpenSearch_Hybrid_Index.ipynb               ← Connector Ollama BGE-M3 (ML Commons) + ingest pipeline + índice kNN/BM25 sobre 1.100 acórdãos TCU 2026
│   ├── LAB2_Search_Pipeline_RRF.ipynb                   ← Pipelines RRF e Min-Max (busca híbrida com neural query server-side)
│   ├── LAB3_Hybrid_Search_Juridico.ipynb                ← Avaliação MRR/Recall/NDCG sobre 20 queries TCU 2026, registro no LangFuse
│   ├── LAB4_Neural_Sparse_Search.ipynb                  ← SPLADE multilíngue pré-treinado (amazon/neural-sparse/opensearch-neural-sparse-encoding-multilingual-v1)
│   ├── LAB5_Contextual_Retrieval.ipynb                  ← #T09: pré-processar chunks com Groq (fallback Ollama), medir Context Precision/Recall (RAGAS)
│   └── LAB6_LangFuse_Dashboard_Comparativo.ipynb        ← Dashboard 5-vias: BM25 vs Dense vs Hybrid RRF vs Neural Sparse vs Hybrid+Contextual; Δ vs BM25 e RAGAS
│
├── exemplos/
│   ├── EXEMPLO1_Hybrid_Query_Basico.ipynb               ← Referência rápida: hybrid query com BGE-M3 via Ollama (dim=1024)
│   ├── EXEMPLO2_RRF_vs_Normalization.ipynb              ← Comparação RRF × min-max normalization
│   └── EXEMPLO3_Conversational_RAG_Pipeline.ipynb       ← Conversational RAG com Groq + Ollama BGE-M3
│
└── datasets/
    ├── corpus_juridico_aula4.json                       ← Corpus legado: 20 docs (compatibilidade)
    ├── corpus_juridico_aula4_v2.json                    ← Corpus enriquecido: 1.100 acórdãos TCU 2026 (extraídos de aula2/datasets/acordao-completo-2026.csv)
    └── queries_avaliacao_aula4.json                     ← 20 queries com ground-truth (top-5 docs relevantes por query)
```

---

## Roteiro da Aula (5 horas)

| Bloco | Duração | Tipo | Conteúdo | Arquivo |
|---|---|---|---|---|
| **1. Revisão + Motivação** | 15 min | Teoria | Limitações da busca puramente vetorial ou BM25 em textos jurídicos | `teoria/AULA4_TEORIA.md §1` |
| **2. Hybrid Search — Arquitetura** | 30 min | Teoria | kNN dense + BM25 sparse, score fusion, search pipelines | `teoria/AULA4_TEORIA.md §2–3` |
| **3. LAB 1 — Índice Híbrido + Connector Ollama** | 45 min | Prática | Connector ML Commons → Ollama BGE-M3, ingest pipeline e índice (1024d) | `labs/LAB1_OpenSearch_Hybrid_Index.ipynb` |
| **4. RRF e Score Fusion — Teoria** | 20 min | Teoria | Reciprocal Rank Fusion, min-max normalization, arithmetic combination | `teoria/AULA4_TEORIA.md §4` |
| **5. LAB 2 — Search Pipeline RRF/Min-Max** | 40 min | Prática | Criar/ativar search pipelines com normalization processor | `labs/LAB2_Search_Pipeline_RRF.ipynb` |
| **6. LAB 3 — Busca Híbrida Jurídica + LangFuse** | 40 min | Prática | Avaliar MRR/Recall/NDCG em 20 queries TCU 2026 + dashboard LangFuse | `labs/LAB3_Hybrid_Search_Juridico.ipynb` |
| **7. Neural Sparse — Teoria** | 15 min | Teoria | SPLADE, sparse neural vectors, eficiência vs. dense | `teoria/AULA4_TEORIA.md §5` |
| **8. LAB 4 — Neural Sparse Multilíngue** | 30 min | Prática | Registrar modelo pré-treinado, ingest/search via neural_sparse | `labs/LAB4_Neural_Sparse_Search.ipynb` |
| **9. Contextual Retrieval — Teoria** | 15 min | Teoria | Pré-processamento de chunks com contexto, impacto em Context Precision | `teoria/AULA4_TEORIA.md §6` |
| **10. LAB 5 — Contextual Retrieval c/ Groq** | 45 min | Prática | Enriquecer 100 chunks via Groq (fallback Ollama), medir RAGAS antes/depois | `labs/LAB5_Contextual_Retrieval.ipynb` |
| **11. LAB 6 — Dashboard 5-vias** | 30 min | Prática | Comparativo BM25 vs Dense vs Hybrid vs Neural Sparse vs Hybrid+Contextual | `labs/LAB6_LangFuse_Dashboard_Comparativo.ipynb` |

---

## Objetivos de Aprendizagem (conforme ementa)

Ao final desta aula, o aluno será capaz de:

1. **Explicar** a diferença entre busca densa (kNN), esparsa (BM25 e neural sparse) e híbrida em contexto jurídico
2. **Conectar o OpenSearch ao Ollama** via Connector ML Commons e indexar com **embeddings server-side** (BGE-M3, dim=1024)
3. **Configurar search pipelines** com normalization processor e RRF
4. **Comparar estratégias de fusão de scores** (RRF vs. min-max normalization) com métricas objetivas
5. **Implementar Neural Sparse Search** com o modelo pré-treinado multilíngue `opensearch-neural-sparse-encoding-multilingual-v1`
6. **Aplicar Contextual Retrieval (#T09)** — enriquecer chunks via Groq (fallback Ollama) e medir impacto via RAGAS
7. **Quantificar a melhoria do RAG** com cada técnica e registrar tudo no **LangFuse** (Scores API)

---

## Stack Tecnológico

| Componente | Ferramenta | Papel no Pipeline |
|---|---|---|
| Motor de busca | **OpenSearch 3.x** | Índice híbrido (kNN + BM25), neural search, neural_sparse, search pipelines |
| Plugin Neural | **Neural Search Plugin** | Queries `neural` e `neural_sparse` integradas a modelos do ML Commons |
| Plugin ML | **ML Commons Plugin** | Connector remoto (Ollama) + modelo pré-treinado SPLADE multilíngue |
| Embeddings densos | **Ollama `bge-m3`** (dim=1024) | Embeddings multilíngues via connector remoto (server-side) |
| Neural Sparse | **`amazon/neural-sparse/opensearch-neural-sparse-encoding-multilingual-v1`** | Sparse encoding via ingest/search pipelines `neural_sparse` |
| Score Fusion | **RRF + min-max normalization** | Combinação de scores BM25 + neural (search pipelines) |
| LLM | **Groq `llama-3.1-8b-instant`** (primário) · **Ollama `llama3.2:3b`** (fallback) | Contextualização de chunks no LAB5; geração no exemplo conversacional |
| Avaliação | **RAGAS** | Context Precision/Recall antes/depois do Contextual Retrieval |
| Observabilidade | **LangFuse** | Scores agregados, spans por query/estratégia, Δ vs BM25, dashboards |

---

## Fichas de Técnicas RAG — Esta Aula

### Ficha T04 — Hybrid Search

| Campo | Conteúdo |
|---|---|
| **ID** | #T04 |
| **Categoria** | Retrieval Avançado |
| **Subtítulo** | Fusão de busca vetorial e lexical |
| **Descrição** | Combina busca densa (BGE-M3 via Ollama connector) e busca esparsa (BM25/TF-IDF). Scores normalizados e fundidos via RRF ou arithmetic combination. Captura tanto semântica quanto match exato — essencial em textos jurídicos com terminologia técnica. |
| **Aplicabilidades** | Pesquisa em legislação, jurisprudência e doutrina; sistemas de compliance; investigação policial; portais de e-gov; acórdãos TCU |
| **Vantagens** | Melhor Recall e MRR que abordagem única; robusto para variações de vocabulário jurídico |
| **Limitações** | Latência maior; requer tuning de pesos; dependência de infraestrutura |
| **Lab** | LAB1 (índice + connector) + LAB2 (pipeline RRF) + LAB3 (avaliação) |
| **Referência** | CHEN et al. (2022). arXiv:2210.11934; OpenSearch Docs, 2026. |

### Ficha — Neural Sparse Search (SPLADE Multilíngue)

| Campo | Conteúdo |
|---|---|
| **Categoria** | Retrieval Avançado |
| **Subtítulo** | Representação esparsa neural (SPLADE) |
| **Descrição** | Usa o modelo pré-treinado `opensearch-neural-sparse-encoding-multilingual-v1` no ML Commons para gerar vetores esparsos token→peso, indexados como `rank_features`. Combina interpretabilidade do BM25 com compreensão semântica dos transformers. |
| **Aplicabilidades** | Pesquisa em grandes corpora legislativos; busca de precedentes; integração com sistemas legacy de índice invertido |
| **Vantagens** | Interpretabilidade (termos com pesos); eficiência computacional; qualidade próxima ao dense retrieval; multilíngue out-of-the-box |
| **Limitações** | Ingestão mais lenta (encoding pesado); modelo único por idioma |
| **Lab** | LAB4 (registro do modelo + ingest/search via `neural_sparse`) |
| **Referência** | FORMAL et al. (2021). SPLADE. arXiv:2107.05720; OpenSearch Docs (Neural Sparse with Pipelines). |

### Ficha T09 — Contextual Retrieval

| Campo | Conteúdo |
|---|---|
| **ID** | #T09 |
| **Categoria** | Retrieval Avançado |
| **Subtítulo** | Enriquecimento contextual de chunks via Groq + Ollama |
| **Descrição** | Antes da indexação, cada chunk é enriquecido com um trecho de contexto situacional gerado pelo Groq `llama-3.1-8b-instant` (fallback Ollama `llama3.2:3b`). O prompt descreve o papel do chunk no acórdão TCU. O chunk enriquecido é reindexado, melhorando Context Precision/Recall (RAGAS). |
| **Aplicabilidades** | Acórdãos TCU/STJ extensos; laudos periciais multi-fase; procedimentos policiais |
| **Vantagens** | Melhora significativa em Context Precision; sem mudança na arquitetura de retrieval; Groq oferece latência ~10× menor que vLLM em CPU |
| **Limitações** | Custo de pré-processamento (1 chamada LLM por chunk); rate limit do Groq (mitigado pelo fallback automático para Ollama) |
| **Lab** | LAB5 (100 chunks, RAGAS antes/depois, scores no LangFuse) |
| **Referência** | ANTHROPIC. *Introducing Contextual Retrieval*. 2024. <https://www.anthropic.com/news/contextual-retrieval>. |

---

## Datasets da Aula 4

- **`corpus_juridico_aula4_v2.json`** (4,4 MB): 1.100 acórdãos do TCU de 2026 extraídos de `aula2/datasets/acordao-completo-2026.csv`. Campos: `id`, `tipo`, `titulo`, `conteudo` (sumário + assunto + acórdão + relatório, limpos de HTML), `metadata` (numAcordao, ano, relator, colegiado, dataSessao, interessados, entidade).
- **`queries_avaliacao_aula4.json`**: 20 queries temáticas com 4–5 documentos relevantes por query (ground-truth derivado de overlap de keywords no conteúdo).
- **`corpus_juridico_aula4.json`** (legado): 20 docs sintéticos — mantido para retrocompatibilidade. Os LABs preferem `_v2` automaticamente.

---

## Avaliação

Ver `AVALIACAO_AULA4.md` para rubricas completas.

| Entregável | Peso | Lab |
|---|---|---|
| Índice híbrido funcional + connector Ollama BGE-M3 | 20% | LAB1 |
| Search pipelines RRF e Min-Max configurados | 15% | LAB2 |
| Análise comparativa MRR/Recall/NDCG no LangFuse | 20% | LAB3 |
| Neural Sparse Search com modelo pré-treinado funcional | 15% | LAB4 |
| Contextual Retrieval com Δ RAGAS positivo registrado | 20% | LAB5 |
| Dashboard 5-vias com melhoria do RAG vs BM25 (Δ %) | 10% | LAB6 |

---

## Referências Bibliográficas (ABNT)

ANTHROPIC. **Introducing Contextual Retrieval**. Blog Anthropic, 2024. Disponível em: <https://www.anthropic.com/news/contextual-retrieval>.

CHEN, J. et al. **BGE M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity Text Embeddings**. arXiv:2309.07597, 2024.

CHEN, Y. et al. **Out-of-Domain Semantics to the Rescue! Zero-Shot Hybrid Retrieval Models**. arXiv:2210.11934, 2022.

CORMACK, G. V.; CLARKE, C. L. A.; BUETTCHER, S. **Reciprocal Rank Fusion Outperforms Condorcet**. SIGIR Forum, v. 43, n. 1, p. 1-8, 2009.

ES, S. et al. **RAGAS: Automated Evaluation of Retrieval Augmented Generation**. arXiv:2309.15217, 2023.

FORMAL, T. et al. **SPLADE: Sparse Lexical and Expansion Model for First Stage Ranking**. arXiv:2107.05720, 2021.

GROQ INC. **Groq API Documentation**. Disponível em: <https://console.groq.com/docs>.

LANGFUSE. **Scores API Documentation**. Disponível em: <https://langfuse.com/docs/scores>.

LOMBARDO, A. **Integrating Ollama and OpenSearch for Vector Indexing Using Models from Ollama**. LinkedIn Pulse, 2024. Disponível em: <https://www.linkedin.com/pulse/integrating-ollama-opensearch-vector-indexing-using-models-lombardo-anmie/>.

OLLAMA. **BGE-M3 Model**. Disponível em: <https://ollama.com/library/bge-m3>.

OPENSEARCH PROJECT. **Hybrid Search**. 3.0 Docs. Disponível em: <https://docs.opensearch.org/3.0/vector-search/ai-search/hybrid-search/>.

OPENSEARCH PROJECT. **Neural Sparse Search**. 3.0 Docs. Disponível em: <https://docs.opensearch.org/3.0/vector-search/ai-search/neural-sparse-search/>.

OPENSEARCH PROJECT. **Neural Sparse Search Using Pipelines**. 3.0 Docs. Disponível em: <https://docs.opensearch.org/3.0/vector-search/ai-search/neural-sparse-with-pipelines/>.

OPENSEARCH PROJECT. **Connecting to Externally Hosted Models**. 3.0 Docs. Disponível em: <https://docs.opensearch.org/3.0/ml-commons-plugin/remote-models/index/>.

TRIBUNAL DE CONTAS DA UNIÃO (TCU). **Acórdãos 2026 — base completa**. Brasília: TCU, 2026.
