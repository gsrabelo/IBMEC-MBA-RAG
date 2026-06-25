# Estudo inicial — Extração de dados para RAG (documentos complexos, tabelas e dados estruturados)

**Curso:** MBA em RAG & CAG Aplicados a Direito e Segurança Pública · Aula 6 (Indexação Avançada)
**Status:** estudo exploratório (panorama). **Nada implementado** — material para avaliarmos a abordagem antes de montar labs.
**Recorte:** preferência por ferramental **open source**, conectável ao stack do curso (Haystack + OpenSearch + Ollama + Groq).
**Data da pesquisa:** junho/2026.

---

## 1. Por que extração é o gargalo do RAG (e mais ainda no jurídico)

A qualidade de um sistema RAG é limitada pela qualidade do que entra no índice: "garbage in, garbage out". No nosso domínio isso é crítico porque os documentos são **heterogêneos e difíceis**: acórdãos com numeração e notas de rodapé, PDFs **escaneados** (digitalizações de processos), **tabelas** de execução orçamentária, anexos em **planilhas** (XLSX/CSV), portarias em DOCX, e-mails e ofícios. Um detalhe recorrente em produção: os divisores de texto padrão **picam as linhas de uma tabela entre chunks diferentes**, e o LLM recebe colunas fragmentadas e cabeçalhos desalinhados em vez de dados estruturados — uma das maiores fontes de erro factual em pipelines reais.

O problema se divide em três naturezas de dado, que pedem ferramentas diferentes:

- **Não-estruturado** — PDF nativo, PDF escaneado, DOCX, HTML, e-mail, imagens. Precisa de *parsing de layout* + OCR.
- **Semi-estruturado** — tabelas *dentro* de documentos. Precisa de *table detection/structure recognition*.
- **Estruturado** — planilhas (XLSX/CSV) e bancos de dados. Pede *serialização inteligente* ou *consulta* (text-to-SQL/semantic layer), não embedding ingênuo.

---

## 2. Documentos não-estruturados: as três famílias de parsers

### 2.1 Pipelines de layout "clássicos" (modelos especializados encadeados)

Detectam regiões da página (título, parágrafo, tabela, figura) com modelos de layout e depois reconhecem o conteúdo de cada uma. Boa preservação de estrutura, rodam local, saída em Markdown/JSON.

- **Docling (IBM Research)** — já usado na nossa Aula 2. Usa **DocLayNet** (layout) + **TableFormer** (estrutura de tabelas), tem API Python + CLI e integração nativa com **Haystack**, LangChain e LlamaIndex. É a opção open source "meio-termo" mais recomendada para stacks self-hosted com bom equilíbrio qualidade/controle.
- **Unstructured** (`unstructured-io`) — converte PDF/e-mail/HTML/Office em elementos rotulados (título, narrativa, tabela…). Muito popular e prático para ingestão genérica, mas relatos recentes apontam **queda de acurácia em layouts complexos** — usar com cautela e validar no nosso corpus.
- **Marker** (`marker-pdf`) — gera Markdown de alta qualidade, foco em RAG self-hosted; bom em PDFs nativos.

### 2.2 Parsers VLM "end-to-end" — a virada de 2025

Em vez de encadear modelos, um único **Vision-Language Model** lê a imagem da página e produz Markdown/JSON com layout, tabelas e ordem de leitura. Saltaram para o estado da arte em 2025, com melhor robustez em layouts difíceis e multilíngue, ao custo de **GPU** e do risco de *alucinação* (o modelo "inventa" texto que não está na página).

- **MinerU 2 (OpenDataLab)** — PDF/imagem/DOCX/PPTX/**XLSX** → Markdown/JSON, detecção de 109 idiomas, visualização de layout. Forte para pipelines de retrieval.
- **olmOCR 2 (Allen AI)** — VLM 7B (sobre Qwen2.5-VL) treinado com RLVR, ~250k páginas; alta vazão preservando ordem natural de leitura. Bom para volume.
- **dots.ocr (rednote-hilab)** — VLM compacto (~1.7B), licença MIT, layout + reconhecimento num só modelo, **SOTA em texto/tabelas/ordem de leitura** no OmniDocBench com fórmulas comparáveis a modelos maiores. Ótimo custo-benefício.
- **GOT-OCR 2.0 / Nougat** — fortes em conteúdo científico (LaTeX, fórmulas, tabelas tipografadas).

### 2.3 OCR puro (para escaneados)

Quando o PDF é imagem (digitalização de processo físico), entra o OCR antes do parsing:

- **Surya** — OCR multilíngue moderno, com detecção de layout e tabelas; bom como camada OCR.
- **PaddleOCR** — maduro, rápido, muito usado em produção.
- **Tesseract** — clássico, leve, qualidade inferior em documentos ruins; serve de baseline.

### 2.4 Referência (proprietárias, fora do recorte OSS)

Úteis como "teto de qualidade" para comparar: **LlamaParse** (forte em layouts complexos e saída pronta para LLM), **Reducto**, **Azure Document Intelligence**, **AWS Textract**. Cito apenas como balizamento — o foco do estudo é OSS.

---

## 3. Tabelas: o ponto mais doloroso

Há três abordagens, frequentemente combinadas:

- **Regra/heurística (determinístico, sem GPU):** **Camelot** (modo *lattice* é o melhor para tabelas **com borda**; *stream* para sem borda), **pdfplumber** (fallback confiável), **PyMuPDF**, **Tabula** (Java). Previsíveis e baratas, mas frágeis em tabelas irregulares/mescladas.
- **ML / Transformer:** **Table Transformer (TATR)** para detecção+estrutura, **TableFormer** (dentro do Docling), **GMFT**. Estudos comparativos mostram que o TATR é mais **versátil e consistente** entre categorias de documento, enquanto Camelot ganha em tipos específicos (tabelas com borda nítida).
- **VLM:** os parsers da seção 2.2 já entregam a tabela como Markdown/HTML/JSON diretamente.

**Estratégias de chunking de tabela (decisivas para o retrieval):**

1. **Não picar a tabela** — tratá-la como uma unidade; serializar para **Markdown/HTML/JSON** preservando cabeçalhos.
2. **Resumo + original (multi-vector):** gerar um resumo em linguagem natural da tabela, **embeddar o resumo** (casa melhor com perguntas) e **devolver a tabela original** ao LLM. É o padrão *multi-vector retriever* (resumo→conteúdo), que conecta direto com Parent-Child/RAPTOR (seção 6).
3. **Linha-a-linha com contexto:** para tabelas grandes, cada linha vira um registro com cabeçalho repetido + contexto da tabela (parente disso é o *Contextual Retrieval* da Aula 4).

---

## 4. Dados estruturados (planilhas e bancos): dois caminhos

Embeddar células de planilha como texto quase nunca funciona para perguntas **numéricas/agregadas** ("qual o total de multas em 2024?"). Há dois caminhos, escolhidos pela natureza da pergunta:

- **(a) RAG sobre tabelas (busca semântica):** bom para perguntas **qualitativas/lookup** ("o que diz a linha sobre o contrato X?"). Serializa linhas/tabelas (seção 3) e indexa. Simples, encaixa no pipeline atual.
- **(b) Text-to-SQL / Semantic Layer / agente com ferramenta:** bom para **agregações, filtros e cálculos**. A planilha/CSV vira tabela (DuckDB/SQLite/Postgres) e o LLM **gera SQL** sobre um *semantic layer* (mapa de termos de negócio → colunas). Mais preciso para números, mas exige modelar esquema e validar o SQL gerado. Aqui o RAG vira **RAG agêntico** (uma das técnicas das aulas finais).

Boa prática emergente: **roteador** que decide (a) vs (b) conforme a intenção da pergunta — encaixa perfeitamente no que já fizemos na Aula 8 com o `ConditionalRouter`/CRAG.

---

## 5. Retrieval visual/multimodal (quando o parsing falha ou a página "é visual")

- **ColPali / ColQwen (late interaction sobre a imagem da página):** em vez de extrair texto, embeddam **patches da imagem** da página e fazem *late interaction* (MaxSim, herdado do ColBERT). Dispensam parsing, são robustos a layouts complexos/figuras e estão no SOTA de *visual document retrieval* (ViDoRe). Custo: índices multi-vetor maiores e necessidade de VLM. Trilha promissora para acórdãos escaneados onde o OCR sofre.
- **Multimodal RAG (texto + tabela + imagem):** dois padrões consolidados — (i) **embeddings multimodais** de imagem + busca direta + LLM multimodal na geração; (ii) **resumir** cada imagem/tabela com um VLM, embeddar os resumos e recuperar (mais barato, é o *multi-vector retriever* do LangChain). Para o nosso caso, o padrão (ii) é o mais pragmático.

---

## 6. Como isso se conecta com a indexação avançada que já temos

Este é o elo que você pediu — a extração não substitui as técnicas das Aulas 4/6, ela **alimenta** elas:

- **Parent-Child (Aula 6):** tabela/figura inteira = **pai** (contexto na resposta); resumo ou linhas = **filhos** (precisão na busca). Mesmo mecanismo de auto-merge.
- **Contextual Retrieval (Aula 4):** prefixar a cada chunk de tabela/linha um contexto ("Tabela X do acórdão Y, exercício 2024…") antes de embeddar.
- **RAPTOR (Aula 6):** resumos hierárquicos sobre **seções e tabelas** do documento — perguntas amplas casam com o resumo, específicas com a folha.
- **Busca híbrida BM25 + densa (Aula 4):** essencial para **números, códigos e siglas** de tabelas (ex.: "Acórdão 1234/2024"), onde o léxico exato importa.
- **Metadados de extração:** página, `tipo` (tabela/figura/texto), fonte, exercício — viram **filtros** no OpenSearch e melhoram precisão e auditabilidade (alinhado ao requisito jurídico de rastreabilidade).

Ou seja: a "camada de extração" entra **antes** do que já construímos, produzindo chunks de melhor qualidade e metadados ricos.

---

## 7. Como avaliar (benchmarks e métricas)

- **OmniDocBench (CVPR 2025, OpenDataLab):** benchmark de referência para *document parsing*, com 9 tipos de fonte (artigos, livros, manuscritos, jornais densos…) e anotações finas; avalia Surya, GOT-OCR, Mathpix e VLMs (Qwen2-VL, InternVL2, GPT-4o). Atenção: já é considerado **saturado** pelos modelos de ponta — útil como referência, mas convém **medir no nosso próprio corpus** (acórdãos/planilhas reais).
- **Métricas práticas:** *edit distance*/CER para texto; **TEDS** (Tree-Edit-Distance Similarity) para estrutura de tabelas; acerto de **ordem de leitura**; e, no fim, a métrica que importa para nós — **impacto no RAG** (Recall@k e Faithfulness via RAGAS, como já fazemos nas Aulas 5/8).

---

## 8. Trade-offs para a decisão

| Eixo | Determinístico (Camelot/pdfplumber/Docling) | VLM end-to-end (MinerU/dots.ocr/olmOCR) | Visual retrieval (ColPali) |
|---|---|---|---|
| Qualidade em layout difícil | média | alta | alta (sem extrair texto) |
| Custo/infra | baixo (CPU) | alto (GPU) | alto (GPU + índice multi-vetor) |
| Previsibilidade | alta | média (risco de alucinação) | média |
| Escaneados ruins | fraco (precisa OCR) | bom | bom |
| Números/códigos exatos | bom | bom | depende |
| Encaixe no stack atual | imediato (Haystack) | via conversão p/ Markdown→chunks | exige retriever multi-vetor |

---

## 9. Recomendação inicial (para discutirmos)

Uma arquitetura **em camadas**, toda OSS e plugável no pipeline atual:

1. **Roteador de ingestão por tipo de arquivo:** PDF nativo / PDF escaneado / DOCX-HTML / XLSX-CSV seguem caminhos distintos.
2. **Parsing de não-estruturado:** **Docling como baseline** (continuidade com a Aula 2), comparado contra **MinerU 2** e **dots.ocr** nos casos difíceis (escaneados e tabelas). Surya/PaddleOCR como camada OCR quando o PDF é imagem.
3. **Tabelas:** Camelot (lattice) + pdfplumber para tabelas com borda; TableFormer (via Docling) / TATR para o resto; **chunking "tabela inteira + resumo" (multi-vector)**.
4. **Planilhas/estruturado:** começar por **serialização + busca semântica**; evoluir para **text-to-SQL (DuckDB)** quando aparecerem perguntas de agregação numérica.
5. **(Trilha opcional/avançada) ColPali** para um experimento de *visual retrieval* em acórdãos escaneados.
6. **Sempre:** metadados ricos (página, tipo, fonte) + ligação com Parent-Child/RAPTOR/Contextual/híbrido já existentes, e **avaliação no nosso corpus** com RAGAS.

**PoC sugerida para avaliar antes de comprometer:** pegar ~20 documentos representativos (alguns nativos, alguns escaneados, alguns com tabela, 1–2 planilhas), passar por Docling vs MinerU vs dots.ocr, medir TEDS nas tabelas e Recall@k/Faithfulness no RAG, e decidir o ferramental por tipo de dado.

---

## 10. Riscos e considerações

- **Alucinação de VLM:** parsers VLM podem "inventar" texto ausente — perigoso no jurídico; mitigar com verificação e preferir determinístico onde a tabela for limpa.
- **GPU/custo:** os VLMs SOTA exigem GPU; avaliar se roda local (Ollama/vLLM) ou se vale uma trilha CPU-only (Docling/Camelot).
- **PII e sigilo:** documentos de segurança pública contêm dados sensíveis; manter tudo on-premise (coerente com a escolha de OpenSearch/Ollama local).
- **Escaneados de baixa qualidade:** OCR ruim degrada tudo a jusante; ter etapa de avaliação de qualidade da digitalização.
- **Manutenção:** o ecossistema muda rápido (vários modelos de 2025); fixar versões e reavaliar periodicamente.

---

## 11. Próximos passos sugeridos (se aprovarmos a abordagem)

1. Montar a **PoC comparativa** (seção 9) e medir no nosso corpus.
2. Definir o **roteador de ingestão** e os parsers por tipo.
3. Criar 1–2 **labs**: (a) "extração de documentos complexos + tabelas → chunks com metadados → Parent-Child/RAPTOR"; (b) "planilha → serialização vs text-to-SQL".
4. Opcional: lab de **visual retrieval (ColPali)** como técnica de fronteira.

---

## Fontes

- [Best AI Document Parsers for 2025 — LlamaIndex](https://www.llamaindex.ai/insights/document-parser-comparison-2025)
- [Best PDF Parsers for AI and RAG Workflows in 2026 — Firecrawl](https://www.firecrawl.dev/blog/best-pdf-parsers)
- [Docling vs LlamaParse vs Unstructured vs Reducto — Reducto](https://llms.reducto.ai/document-parser-comparison)
- [PDF Data Extraction Benchmark 2025 (Docling, Unstructured, LlamaParse) — Procycons](https://procycons.com/en/blogs/pdf-data-extraction-benchmark/)
- [OmniDocBench (CVPR 2025) — GitHub OpenDataLab](https://github.com/opendatalab/OmniDocBench) · [arXiv 2412.07626](https://arxiv.org/abs/2412.07626)
- [OmniDocBench is Saturated, What's Next for OCR Benchmarks? — LlamaIndex](https://www.llamaindex.ai/blog/omnidocbench-is-saturated-what-s-next-for-ocr-benchmarks)
- [Beyond Text Extraction: The 2025 Open OCR Revolution (VLMs) — Medium](https://atul4u.medium.com/beyond-text-extraction-the-2025-open-ocr-revolution-powered-by-vision-language-models-89ad33d36bbf)
- [Supercharge your OCR Pipelines with Open Models — Hugging Face](https://huggingface.co/blog/ocr-open-models)
- [dots.ocr — GitHub rednote-hilab](https://github.com/rednote-hilab/dots.ocr) · [arXiv 2512.02498](https://arxiv.org/html/2512.02498v1)
- [MinerU — OpenDataLab](https://opendatalab.github.io/MinerU/)
- [Camelot — PDF Table Extraction](https://camelot-py.readthedocs.io/) · [gmft — PyPI](https://pypi.org/project/gmft/0.2.0rc0/)
- [A Comparative Study of PDF Parsing Tools (TATR vs rule-based) — arXiv](https://arxiv.org/html/2410.09871v1)
- [Build RAG with Tables: PDFs and Excel (2026) — Markaicode](https://markaicode.com/rag-tables-pdf-excel-extraction/)
- [Multi-Vector Retriever for RAG on tables, text and images — LangChain](https://blog.langchain.com/semi-structured-multi-modal-rag/)
- [An Overview of Late Interaction Retrieval (ColBERT, ColPali, ColQwen) — Weaviate](https://weaviate.io/blog/late-interaction-overview)
- [ColPali: Efficient Document Retrieval with Vision Language Models — arXiv 2407.01449](https://arxiv.org/abs/2407.01449)
- [Semantic layer — Wikipedia](https://en.wikipedia.org/wiki/Semantic_layer)

*Observação: parte das fontes são posts técnicos/curadorias; os pontos centrais foram triangulados entre o benchmark acadêmico (OmniDocBench/CVPR 2025), repositórios oficiais e múltiplas curadorias independentes. Recomenda-se validar números/qualidade no nosso próprio corpus antes de decidir.*
