# Scripts da Aula 10 — Guia de Uso

Scripts Python simples, de linha de comando, sobre **Agentic RAG** e **Adaptive RAG**
(padrão **ReAct**), usando o componente **`Agent`** do **Haystack**.

Stack: **Haystack** (Agent/Tool/ConditionalRouter) · **Groq** (LLM com tool-calling) ·
**Ollama** (embeddings) · **OpenSearch** (jurisprudência) · **SQLite** (dados
estruturados) · **Tavily** (web, opcional) · **LangFuse** (observabilidade).

> **Haystack atende a Aula 10 por completo.** O roteiro original usa LangGraph, mas o
> Haystack tem `Agent` (laço ReAct/tool-calling nativo), `Tool`/`ToolInvoker` e
> `ConditionalRouter` — cobrindo Agentic RAG e Adaptive RAG sem trocar de stack. Teoria
> completa em [`../teoria/AULA10_TEORIA.md`](../teoria/AULA10_TEORIA.md).

---

## 1. As três ferramentas do agente

| Ferramenta | O que faz | Backend |
|---|---|---|
| `buscar_documentos` | busca semântica na jurisprudência | OpenSearch (índice `aula4_hibrido`) + Ollama |
| `buscar_web` | fatos recentes na web | Tavily (com *fallback* offline) |
| `consultar_banco` | dados estruturados via **text-to-SQL** (só `SELECT`) | SQLite `juridico_segpub.db` |

O **agente** (Haystack `Agent`) decide sozinho quais usar e em que ordem (ReAct). O
**Adaptive RAG** classifica a complexidade da pergunta e roteia para 3 caminhos
(sem retrieval / 1 busca / agente), controlando custo e latência.

---

## 2. Pré-requisitos

1. **Python 3.10+** com o ambiente virtual do curso ativado.
2. **Ollama** com o modelo de embedding: `ollama pull nomic-embed-text`.
3. **Chave da Groq** no `.env` (`GROQ_API_KEY`). Modelo: `llama-3.3-70b-versatile`
   (tool-calling). **Não use gpt-oss** (reasoning) — o tool-calling fica instável.
4. **Índice da Aula 4** populado (`aula4_hibrido`) para a ferramenta `buscar_documentos`.
5. (Opcional) `TAVILY_API_KEY` para a `buscar_web`.

```bash
pip install -r requirements.txt
```

---

## 3. Ordem recomendada e como usar cada script

Rode tudo de dentro da pasta `aula10/scripts`.

### `00_check_ambiente.py` — confere se está tudo pronto
```bash
python 00_check_ambiente.py --testar-groq
```

### `01_preparar_dados.py` — cria o SQLite + checa o índice
Cria `juridico_segpub.db` (acórdãos, ocorrências, legislação, doutrina) e verifica o
índice `aula4_hibrido`.
```bash
python 01_preparar_dados.py
python 01_preparar_dados.py --recriar
```

### `02_react_manual.py` — o ciclo ReAct "na unha" (didático)
Mostra `Thought → Action → Observation` passo a passo, sem o componente Agent.
```bash
python 02_react_manual.py --pergunta "quantas ocorrencias de estelionato em SP e o que a lei diz?"
python 02_react_manual.py --pergunta "o que e habeas corpus?"
```

### `03_agente_ferramentas.py` — Agentic RAG com o `Agent` do Haystack
O agente decide as ferramentas; imprime a **trajetória** (ferramentas chamadas) e a
resposta. Com LangFuse, gera o trace `agente-aula10`.

Tem um **fallback corretivo para a web** (estilo CRAG): se a resposta final vier
**insuficiente** (ex.: "não consta", muito curta) **e** o agente **não** tiver usado a
`buscar_web`, o script aciona a web e refaz a resposta, deixando claro que a fonte é a
web. O agente já é instruído (system prompt) a usar a web como fallback; este check é a
rede de segurança caso ele não o faça. Exige `TAVILY_API_KEY` (senão apenas avisa).
Desligue com `--sem-fallback-web`.
```bash
python 03_agente_ferramentas.py --pergunta "quantas ocorrencias de estelionato em SP e o que a jurisprudencia diz?"
python 03_agente_ferramentas.py --pergunta "decisoes recentes de 2025 sobre o tema X"   # tende a acionar o fallback web
python 03_agente_ferramentas.py --pergunta "..." --sem-langfuse --sem-fallback-web
```

### `04_adaptive_rag.py` — Adaptive RAG (classificador + 3 caminhos)
Classifica a complexidade e roteia: `sem_retrieval` (direto) / `simples` (1 busca) /
`complexa` (agente). Imprime a rota e o tempo.
```bash
python 04_adaptive_rag.py --pergunta "o que e habeas corpus?"
python 04_adaptive_rag.py --pergunta "qual a pena do art. 2 da Lei 12.850?"
python 04_adaptive_rag.py --pergunta "compare prisao preventiva em crimes financeiros vs violentos"
```

### `05_avaliar_custo.py` — distribuição de rotas + custo de 1.000 execuções
Classifica um conjunto de perguntas, mostra a distribuição de rotas e estima o custo —
comparando "tudo agente" vs "adaptativo" (atende o critério de avaliação da aula).
```bash
python 05_avaliar_custo.py --execucoes 1000
```

---

## 4. Resumo de dependências por script

| Script | Haystack | Groq | Ollama | OpenSearch | SQLite | Tavily | LangFuse |
|--------|:--------:|:----:|:------:|:----------:|:------:|:------:|:--------:|
| 00_check_ambiente | ✓ | ✓ | ✓ | ✓ | ✓ | opc. | opc. |
| 01_preparar_dados | — | — | — | ✓ (checa) | ✓ | — | — |
| 02_react_manual | — | ✓ | ✓ | ✓ | ✓ | opc. | — |
| 03_agente_ferramentas | ✓ | ✓ | ✓ | ✓ | ✓ | opc. | opc. |
| 04_adaptive_rag | ✓ | ✓ | ✓ | ✓ | ✓ | opc. | — |
| 05_avaliar_custo | — | ✓ | — | — | — | — | — |

> `_comum.py` não é executado diretamente: configura Groq/Ollama/OpenSearch, define as 3
> ferramentas (`Tool`) e cria o `Agent` (`criar_agente`).

---

## 5. Observações

- **Por que Haystack e não LangGraph:** mantemos um stack único (aulas 1–8). O `Agent`
  do Haystack entrega o laço ReAct com guards (`max_agent_steps`, `exit_conditions`).
- **Guards do agente:** `max_agent_steps=5` (evita loop), e a ferramenta SQL só executa
  `SELECT` (segurança). Ajuste `--max-passos` conforme a pergunta.
- **Adaptive RAG = economia:** o `05` mostra que rotear por complexidade reduz muito o
  custo vs. sempre usar o agente. Os tokens por rota são estimativas — meça no LangFuse e
  ajuste `TOKENS_ROTA`.
- **buscar_web offline:** sem `TAVILY_API_KEY`, a ferramenta devolve um aviso (os scripts
  rodam mesmo assim); o agente segue com as outras ferramentas.
- **Exercício da aula (4ª ferramenta):** adicionar `consultar_legislacao` é fácil — crie
  uma função + um `Tool` em `_comum.ferramentas()` e o agente passa a usá-la.
