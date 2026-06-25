# Aula 11 — Teoria: Técnicas Complementares

**Curso:** MBA em RAG & CAG Aplicados a Direito e Segurança Pública
**Aula:** 11 de 12 · 5h · 30% teoria / 70% prática · Nível complementar/avançado
**Normas:** ABNT NBR 6023:2018 / NBR 10520:2023

Cinco frentes complementares ao pipeline RAG canônico:
**Multimodal RAG**, **Compressão (LLMLingua)**, **ColBERT (late interaction)**,
**Time-Aware RAG** e **DSPy** (otimização automática de prompts).

> **Nota de stack.** Cada técnica usa a ferramenta de referência da sua área (não
> dá para forçar tudo no Haystack): Time-Aware é **OpenSearch puro**; Compressão é
> **LLMLingua**; ColBERT é **RAGatouille**; Multimodal é **CLIP**; otimização é **DSPy**.
> O LLM continua a **Groq** e os embeddings o **Ollama**, como nas aulas anteriores.
> Algumas exigem libs pesadas (torch/faiss) — veja `scripts/requirements.txt`.

---

## Sumário

1. [Multimodal RAG (CLIP) — #T17](#1-multimodal)
2. [Compressão / LLMLingua — #T23](#2-compressao)
3. [ColBERT / Late Interaction — #T20](#3-colbert)
4. [Time-Aware RAG](#4-time-aware)
5. [DSPy — otimização automática](#5-dspy)
6. [Quando usar cada uma](#6-quando)
7. [Referências](#7-referencias)

---

## 1. Multimodal RAG (CLIP) — #T17 {#1-multimodal}

A maioria dos RAG só lê texto, mas laudos, BOs e relatórios trazem **tabelas, gráficos,
plantas e imagens**. O Multimodal RAG indexa também imagens e permite buscá-las por
texto.

**Como funciona:** o **CLIP** (OpenAI, 2021) projeta texto e imagem no **mesmo espaço
vetorial**. Então o embedding da query textual ("organograma da facção") fica próximo do
embedding da imagem certa. Pipeline: Docling extrai texto/tabelas/imagens → texto via
embedder (Ollama/BGE) e imagens via CLIP → índice → na query, embeda a pergunta com CLIP
e recupera imagens por similaridade de cosseno.

```
score(texto_query, imagem) = cos( CLIP_text(query), CLIP_image(imagem) )
```

**Trade-off:** cobre lacunas onde a informação está em imagem/tabela; custo: infraestrutura
(modelos CLIP/torch) e embeddings multimodais ainda em evolução.

No lab (`05_multimodal_clip.py`) geramos imagens-exemplo (mapa de calor, organograma,
tabela, gráfico) e mostramos o CLIP recuperando a imagem certa para cada consulta textual.

---

## 2. Compressão / LLMLingua — #T23 {#2-compressao}

RAG ingênuo concatena todos os chunks e manda ao LLM → **custo alto**, **latência** e o
fenômeno **"lost in the middle"** (o LLM ignora o meio de contextos longos).

O **LLMLingua** (Microsoft, 2023) faz **token pruning por perplexidade**: um modelo
pequeno pontua cada token; tokens **previsíveis** (baixa perplexidade, ex.: "de", "que",
"aos") são removidos; o **núcleo semântico** é preservado. O **LLMLingua-2** treina um
classificador (encoder BERT) para isso — é mais rápido e multilíngue (bom para PT).

```
Original (52 tokens): "O artigo 5º da CF estabelece que todos são iguais perante a lei..."
Comprimido (~16):     "Art.5 CF: todos iguais lei, garantindo vida, liberdade, ... propriedade."
```

**Trade-off:** 2–5× menos tokens de entrada (menos custo/latência) sem reranking
separado; risco de perder informação sutil. Alternativa **abstractiva**: RECOMP (resume
em vez de remover).

No lab (`03_compressao_llmlingua.py`): recuperamos o contexto, comprimimos com LLMLingua-2
e respondemos com a Groq — mostrando tokens antes/depois e a economia.

---

## 3. ColBERT / Late Interaction — #T20 {#3-colbert}

Embeddings comuns (BGE-M3, nomic) são **bi-encoders**: comprimem o documento inteiro em
**um vetor** — eficiente, mas termos irrelevantes "diluem" o vetor e perdem precisão.

O **ColBERT** (Khattab & Zaharia, 2020) guarda **um vetor por token** e calcula a
relevância via **MaxSim**: para cada token da query, pega a maior similaridade com
qualquer token do documento, e soma.

```
Score = Σ_q  max_d  cos(q, d)
```

Isso dá **precisão próxima de cross-encoders com velocidade de bi-encoders** (o motor
**PLAID** comprime os vetores por clustering). Custo: índice 3–5× maior.

A lib **RAGatouille** encapsula o ColBERTv2 numa API simples (`index` / `search`). No lab
(`04_colbert_ragatouille.py`) comparamos ColBERT vs. busca densa (bi-encoder) no mesmo
corpus.

| Método | nDCG@10 típico | Latência | Índice |
|---|---|---|---|
| BM25 | 0.45–0.55 | ~5ms | pequeno |
| Bi-encoder | 0.60–0.70 | ~10ms | médio |
| ColBERT | 0.70–0.80 | ~30ms | grande (3–5×) |
| Cross-encoder | 0.75–0.85 | ~200ms | sem índice |

---

## 4. Time-Aware RAG {#4-time-aware}

No Direito, **leis são revogadas, súmulas canceladas, normas alteradas**. Um RAG sem
consciência temporal pode devolver jurisprudência superada ou portaria revogada — **risco
jurídico grave**.

A solução é penalizar documentos antigos com uma **função de decay** (e/ou filtrar por
vigência):

```
decay(idade) = exp( -ln(2) * max(0, idade_dias - offset) / scale )
   scale  = meia-vida em dias (ex.: 365 = peso cai à metade em 1 ano)
   offset = período de graça sem penalização (ex.: 30 dias)
score_final = score_relevancia * decay(idade)
```

| Documento | Idade | decay (scale=365) |
|---|---|---|
| ontem | 1 dia | ~1.00 |
| 1 ano | 365 dias | 0.50 |
| 2 anos | 730 dias | 0.25 |
| 5 anos | 1825 dias | 0.03 |

É a técnica **mais leve e mais alinhada ao nosso stack**: dá para fazer no **OpenSearch**
nativo (`function_score` com `exp` decay) ou re-ranqueando no Python. No lab
(`02_time_aware.py`) recuperamos por relevância e re-ranqueamos por `relevância × decay`,
com opção de **filtro de vigência** — mostrando como a ordem muda priorizando o atual.

Estratégias: decay suave (scale=730d) para jurisprudência histórica; decay agressivo
(scale=180d) para normas operacionais; *hard filter* só vigentes; boost por `versao_vigente`.

---

## 5. DSPy — otimização automática de prompts {#5-dspy}

Prompts são frágeis e a engenharia manual não é reproduzível. O **DSPy** (Stanford, 2024)
trata o pipeline como um **programa compilável**: você declara **módulos** (ex.:
`ChainOfThought`), uma **métrica** e um **dataset**; o **otimizador** (ex.:
`BootstrapFewShot`) escolhe automaticamente instruções e exemplos few-shot que maximizam
a métrica.

```
Você declara:  ChainOfThought("context, question -> answer") + métrica + 10 exemplos
DSPy compila:  prompt com instrução melhor + few-shot selecionados por bootstrapping
```

Otimizadores: `BootstrapFewShot` (barato), `...WithRandomSearch` (médio), `MIPROv2`
(bayesiano, alto), `COPRO` (só instruções).

No lab (`06_dspy_otimizacao.py`) definimos um RAG com `ChainOfThought`, otimizamos com
`BootstrapFewShot` usando a Groq como LM, e comparamos a resposta **antes e depois** da
compilação.

---

## 6. Quando usar cada uma {#6-quando}

| Técnica | Use quando… | Peso de infra |
|---|---|---|
| **Time-Aware** | vigência/recência importam (quase sempre no Direito) | leve (OpenSearch) |
| **Compressão** | contexto longo, custo/latência alto | médio (small LM) |
| **DSPy** | quer prompt robusto/reproduzível, tem dataset | médio (usa o LLM) |
| **ColBERT** | precisa de alta precisão de retrieval | pesado (torch/faiss) |
| **Multimodal** | informação crítica em imagens/tabelas | pesado (CLIP/torch) |

Não são exclusivas: dá para combinar (ex.: ColBERT + Time-Aware + Compressão num só
pipeline de produção).

---

## 7. Referências {#7-referencias}

JIANG, Huiqiang et al. **LLMLingua: Compressing Prompts for Accelerated Inference of LLMs**.
EMNLP 2023. https://arxiv.org/abs/2310.05736

KHATTAB, Omar; ZAHARIA, Matei. **ColBERT: Efficient and Effective Passage Search via
Contextualized Late Interaction over BERT**. SIGIR 2020. https://arxiv.org/abs/2004.12832

KHATTAB, Omar et al. **DSPy: Compiling Declarative Language Model Calls into Self-Improving
Pipelines**. ICLR 2024. https://arxiv.org/abs/2310.03714

RADFORD, Alec et al. **Learning Transferable Visual Models From Natural Language
Supervision (CLIP)**. ICML 2021. https://arxiv.org/abs/2103.00020

YASUNAGA, Michihiro et al. **Retrieval-Augmented Multimodal Language Modeling**. NeurIPS 2023.

---

*MBA em RAG & CAG Aplicados a Direito e Segurança Pública · Aula 11 de 12*
