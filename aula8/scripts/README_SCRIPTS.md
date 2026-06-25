# Scripts da Aula 8 — Guia de Uso

Scripts Python simples, de linha de comando, sobre **RAG Reflexivo e Auto-Corretivo**:
**Self-RAG (training-free)**, **CRAG (Corrective RAG)** e **roteamento condicional**.

Stack: **Haystack** · **OpenSearch** (vetores) · **Ollama** (embeddings) · **Groq** (LLM/avaliador) · **Tavily** (web search de fallback, opcional) · **LangFuse** (observabilidade).

> **Decisões desta aula:**
> - **Self-RAG e CRAG são pipelines Haystack** — todas as etapas (decisão, avaliador, busca, geração, crítica) viram componentes, traçados no LangFuse.
> - O roteamento condicional do CRAG é feito com o **`ConditionalRouter` do Haystack** (sem LangGraph).
> - O **web search** usa o componente **oficial `TavilyWebSearch`** (pacote `tavily-haystack`) se houver `TAVILY_API_KEY`; senão cai para um **fallback offline** (stub) para os scripts rodarem sempre.
> - Reaproveita o **índice do TCU** da Aula 4 (`aula4_hibrido`); as queries de teste foram **adaptadas à realidade da Aula 8** em `datasets/queries_crag_tcu.json` (umas respondíveis localmente, outras que forçam a rota web).

---

## 1. As três ideias da aula

O RAG comum sempre faz `recuperar → aumentar → gerar`, sem verificar nada. Esta aula
ataca isso de duas formas:

- **Self-RAG** — o LLM decide *quando* recuperar e *audita a própria resposta* com 4
  tokens de controle: `[Retrieve]` (busco ou não?), `[ISREL]` (o documento é
  relevante?), `[ISSUP]` (a resposta tem suporte nos trechos?), `[ISUSE]` (a resposta
  é útil?). Como não temos um modelo com fine-tuning de Self-RAG, fazemos uma versão
  **training-free**: o Groq *emite* esses tokens por prompting.
- **CRAG** — um **avaliador (LLM-as-Judge)** dá uma nota de relevância 0–1 aos
  documentos recuperados e **roteia**: nota alta → usa só o local; nota média →
  funde local + web; nota baixa → usa só a web (Tavily).
- **Roteamento condicional** — implementado com o `ConditionalRouter` do Haystack:
  só a rota não-local dispara o web search.

---

## 2. Pré-requisitos

1. **Python 3.10+** com o ambiente virtual do curso ativado.
2. **OpenSearch** em `localhost:9200` com o índice do TCU (`aula4_hibrido`). Use o
   `01` (reaproveita; se vazio, indexa o corpus do TCU).
3. **Ollama** com o modelo de embedding: `ollama pull nomic-embed-text`.
4. **Chave da Groq** no `.env` (`GROQ_API_KEY`).
5. **Tavily** (opcional): `TAVILY_API_KEY` no `.env` para web search real.
6. **LangFuse** (opcional, para `04`/`06`) — veja o guia da **Aula 3**.

```bash
pip install -r requirements.txt
```

Variáveis opcionais de roteamento no `.env`:
`CRAG_LIMITE_ALTO` (padrão 0.7) e `CRAG_LIMITE_BAIXO` (padrão 0.3).

---

## 3. Ordem recomendada e como usar cada script

Rode tudo de dentro da pasta `aula8/scripts`.

### `00_check_ambiente.py` — confere se está tudo pronto
```bash
python 00_check_ambiente.py --testar-groq
```

### `01_indexar_opensearch.py` — garante/reusa o índice do TCU
```bash
python 01_indexar_opensearch.py            # reaproveita aula4_hibrido se existir
python 01_indexar_opensearch.py --recriar  # reindexa o corpus do TCU
```

### `02_self_rag.py` — Self-RAG training-free (pipeline Haystack)
Pipeline linear: `[Retrieve]` (LLM) → `ConditionalRouter` → (busca + `[ISREL]`) →
montar contexto → geração → `[ISSUP]`/`[ISUSE]`. Cada etapa é um componente, então o
LangFuse captura tudo no trace `self-rag-aula8`. Mostra a decisão `[Retrieve]`, a
relevância `[ISREL]` por documento, a resposta e a autocrítica `[ISSUP]`/`[ISUSE]`.
```bash
python 02_self_rag.py --pergunta "o gestor pode ser multado pelo TCU?"
python 02_self_rag.py --pergunta "o que e responsabilidade civil?"   # tende a [Retrieve=no]
python 02_self_rag.py --pergunta "..." --sem-langfuse
```

### `03_avaliador.py` — o avaliador LLM-as-Judge isolado
Mostra o score 0–1 de cada documento e qual **rota** o CRAG tomaria.
```bash
python 03_avaliador.py --pergunta "quando as contas sao julgadas irregulares?"
python 03_avaliador.py --pergunta "decisoes do STF em 2024 sobre interceptacao"  # score baixo -> web
```

### `04_crag.py` — CRAG completo com ConditionalRouter
Pipeline único: recupera → avalia → **roteia** (local/fusão/web) → (`TavilyWebSearch`) → gera.
Imprime o score, a rota escolhida, as fontes e a resposta (e o trace, se LangFuse ligado).
O `--demo` roda o conjunto curado (`queries_crag_tcu.json`) e mostra a rota de cada query
(esperada vs obtida) — ótimo para ver o roteamento local vs web em ação.
```bash
python 04_crag.py --pergunta "quando as contas sao julgadas irregulares?"
python 04_crag.py --demo                    # roda as queries curadas (local vs web)
python 04_crag.py --pergunta "..." --sem-langfuse
```

### `05_comparar_ragas.py` — CRAG vs Advanced RAG (Faithfulness)
Gera perguntas a partir do próprio índice, responde pelos dois métodos e compara com
**RAGAS** (Faithfulness + ResponseRelevancy).
```bash
python 05_comparar_ragas.py --n 6 --top-k 4
```

### `06_chat_langfuse.py` — chat CRAG instrumentado no LangFuse
Mesmo pipeline do `04`, em loop interativo. Cada pergunta vira um trace `crag-aula8`
com a rota e (quando ocorre) o web search capturados pela auto-instrumentação.
```bash
python 06_chat_langfuse.py
```

---

## 4. Resumo de dependências por script

| Script | OpenSearch | Ollama | Groq | Tavily | LangFuse |
|--------|:----------:|:------:|:----:|:------:|:--------:|
| 00_check_ambiente | ✓ (checa) | ✓ (checa) | ✓ (checa) | ✓ (checa) | ✓ (checa) |
| 01_indexar_opensearch | ✓ | ✓ | — | — | — |
| 02_self_rag | ✓ | ✓ | ✓ | — | — |
| 03_avaliador | ✓ | ✓ | ✓ | — | — |
| 04_crag | ✓ | ✓ | ✓ | opc. | opc. |
| 05_comparar_ragas | ✓ | ✓ | ✓ | opc. | — |
| 06_chat_langfuse | ✓ | ✓ | ✓ | opc. | opc. |

> `_comum.py` e `_componentes.py` não são executados diretamente: o primeiro carrega o
> `.env` e oferece os blocos (busca, avaliador, Tavily, tokens Self-RAG); o segundo traz
> os componentes Haystack do CRAG (`AvaliarRota`, `BuscaWeb`, `MontarContexto`).

---

## 5. Observações

- **Sem Tavily?** Tudo roda: o caminho de web search devolve um aviso "offline" em vez
  de resultados reais. Perguntas sobre fatos recentes (ex.: "STF em 2024") vão pela rota
  `web`/`fusao` e mostram esse aviso — bom para enxergar o roteamento.
- **Self-RAG aqui é training-free:** imita os 4 tokens por prompting. O Self-RAG
  original exige um modelo com fine-tuning específico (ex.: `llama-2-7b-selfrag`).
- **CRAG vs Advanced RAG no RAGAS:** com o índice do TCU respondendo bem às perguntas
  locais, as métricas tendem a ficar próximas; o ganho do CRAG aparece quando o
  retrieval local falha e a rota web/fusão corrige.
- **Limiares do roteador:** ajuste `CRAG_LIMITE_ALTO`/`CRAG_LIMITE_BAIXO` no `.env`.
