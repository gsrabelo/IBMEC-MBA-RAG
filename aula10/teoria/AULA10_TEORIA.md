# Aula 10 — Teoria: Agentic RAG e Adaptive RAG

**Curso:** MBA em RAG & CAG Aplicados a Direito e Segurança Pública
**Aula:** 10 de 12 · 5h · 20% teoria / 80% prática
**Normas:** ABNT NBR 6023:2018 (Referências) / NBR 10520:2023 (Citações)

> **Nota de implementação.** O roteiro original da aula usa LangGraph + LangChain Agents.
> Nos nossos scripts seguimos no **Haystack** (coerência com as aulas 1–8), que **atende a
> totalidade** do que a aula propõe: o componente **`Agent`** implementa o laço ReAct
> (tool-calling), `Tool`/`ToolInvoker` definem as ferramentas, e o **`ConditionalRouter`**
> faz o roteamento do Adaptive RAG. LLM via **Groq** (`llama-3.3-70b-versatile`, com
> *tool calling*), embeddings via **Ollama**, busca no **OpenSearch** e observabilidade no
> **LangFuse** — o mesmo stack das aulas anteriores.

---

## Sumário

1. [Motivação: do RAG linear ao RAG agêntico](#1-motivacao)
2. [O padrão ReAct: Reason + Act](#2-react)
3. [Agentic RAG: o LLM como orquestrador](#3-agentic-rag)
4. [Design de ferramentas (tools)](#4-tools)
5. [Adaptive RAG: roteamento por complexidade](#5-adaptive-rag)
6. [Riscos de agentes e como mitigar](#6-riscos)
7. [Observabilidade e avaliação de agentes](#7-aval)
8. [Mapa: conceito → componente Haystack](#8-mapa)
9. [Aplicações em Direito e Segurança Pública](#9-aplicacoes)
10. [Referências](#10-referencias)

---

## 1. Motivação: do RAG linear ao RAG agêntico {#1-motivacao}

Até aqui (Aulas 2–9) o RAG seguiu, no essencial, um fluxo **fixo**: recuperar →
aumentar → gerar. Mesmo as variações mais sofisticadas (CRAG, Self-RAG, Graph RAG)
decidem *como* recuperar, mas o **plano** é predefinido pelo desenvolvedor.

O **Agentic RAG** dá um passo adiante: é o **LLM que decide**, em tempo de execução,
*quais* ferramentas usar, *em que ordem* e *quantas vezes*, até reunir informação
suficiente para responder. Em vez de um pipeline, temos um **agente** que raciocina e
age em ciclos.

Isso importa no Direito e na Segurança Pública porque muitas perguntas reais são
**multi-etapa**: exigem combinar jurisprudência local, dados estruturados (banco de
ocorrências, legislação) e, às vezes, informação atual da web — e o caminho ótimo só se
revela conforme as respostas parciais chegam.

---

## 2. O padrão ReAct: Reason + Act {#2-react}

O **ReAct** (Yao et al., 2023) intercala dois modos a cada passo:

- **Reasoning (Pensamento):** o LLM raciocina sobre o estado atual e decide a próxima ação.
- **Acting (Ação):** o LLM executa uma ferramenta e **observa** o resultado.

O ciclo se repete até haver informação suficiente:

```
Thought → Action → Observation → Thought → Action → Observation → ... → Answer
```

**Exemplo jurídico (multi-hop):**

```
Pergunta: "Quais precedentes do TCU sobre multa a gestor em contratações emergenciais,
           e há orientação recente de 2025 sobre o tema?"

Thought 1: Preciso dos acórdãos do acervo sobre multa em contratação emergencial.
Action  1: buscar_documentos("multa gestor contratacao emergencial TCU")
Observation 1: [3 acórdãos recuperados]

Thought 2: Falta verificar orientação recente (2025), que pode não estar no acervo.
Action  2: buscar_web("TCU contratacao emergencial 2025 orientacao multa")
Observation 2: [notícias/normas de 2025]

Thought 3: Tenho local + atual; posso responder.
Answer: Com base nos acórdãos [...] e na orientação de 2025 [...].
```

Na prática moderna, o "Thought/Action" do ReAct é realizado via **tool calling** nativo
do LLM: o modelo emite uma *chamada de função* estruturada (nome + argumentos), o
runtime executa a ferramenta e devolve a observação como mensagem. É exatamente o que o
componente **`Agent`** do Haystack faz internamente.

---

## 3. Agentic RAG: o LLM como orquestrador {#3-agentic-rag}

No Agentic RAG, o LLM **planeja** a recuperação. Em vez de uma busca, ele dispõe de um
**conjunto de ferramentas** e escolhe quais usar:

- `buscar_documentos` — busca semântica na jurisprudência (vetorial, OpenSearch);
- `buscar_web` — busca de fatos recentes na web (Tavily);
- `consultar_banco` — consulta a dados **estruturados** (SQLite: acórdãos, ocorrências,
  legislação, doutrina) via *text-to-SQL*.

O agente decide, por exemplo: pergunta conceitual → responde direto; pergunta factual →
uma busca; pergunta comparativa/atual → várias ferramentas em sequência.

---

## 4. Design de ferramentas (tools) {#4-tools}

A qualidade do agente depende **criticamente** da descrição das ferramentas — é por ela
que o LLM decide *quando* usar cada uma.

```text
❌ RUIM:  "Busca documentos"

✅ BOM:   "Busca legislação, jurisprudência e doutrina no banco vetorial local.
          Use quando a pergunta envolver leis, artigos, decisões ou conceitos
          jurídicos. NÃO use para fatos recentes/notícias (use buscar_web).
          Args: query (str) — termos de busca em português."
```

Boas práticas: **nome** claro; **descrição** que diz QUANDO e QUANDO NÃO usar;
**parâmetros tipados** (JSON schema). No Haystack, cada ferramenta é um `Tool(name=...,
description=..., parameters={schema}, function=callable)`.

---

## 5. Adaptive RAG: roteamento por complexidade {#5-adaptive-rag}

O **Adaptive RAG** (Jeong et al., 2024) parte de uma constatação prática: **nem toda
pergunta precisa da mesma estratégia**. Um classificador estima a complexidade da query
e roteia para um de três caminhos:

| Caminho | Quando usar | Custo | Latência |
|---|---|---|---|
| **Sem retrieval** | conhecimento geral basta | mínimo | < 1s |
| **Single-step RAG** | factual, 1 busca resolve | baixo | 2–5s |
| **Multi-step (agente)** | comparativa/multi-fonte | alto | 10–30s |

Exemplos no domínio jurídico:

```
Sem retrieval:   "O que é habeas corpus?"
Single-step:     "Qual o prazo do recurso de apelação criminal?"
Multi-step:      "Compare os precedentes sobre prisão preventiva em crimes financeiros
                  vs. violentos e aponte divergências de 2022–2024."
```

Implementação: um **classificador** (LLM que devolve `sem_retrieval | simples | complexa`)
seguido de um **`ConditionalRouter`** que envia a query ao caminho certo. O caminho
"complexa" aciona o **agente** (ReAct, várias ferramentas); o "simples", uma única busca;
o "sem_retrieval", a geração direta. O ganho é **custo/latência**: você só paga o caminho
caro quando a pergunta exige.

---

## 6. Riscos de agentes e como mitigar {#6-riscos}

| Risco | Causa | Mitigação |
|---|---|---|
| **Loop infinito** | agente nunca "satisfeito" | limite de passos (`max_agent_steps`) |
| **Tool call excessivo** | retrieval redundante | memória de observações / dedup |
| **Custo imprevisível** | muitas chamadas ao LLM | orçamento + alertas + caminho adaptativo |
| **Alucinação de ferramenta** | LLM inventa nome de tool | validação de schema (o runtime rejeita) |
| **Timeout** | ferramenta lenta trava o agente | timeout por tool + fallback |

No contexto jurídico, soma-se a **rastreabilidade**: cada afirmação deve ter origem
verificável (qual ferramenta, qual documento). Os traces do LangFuse ajudam nisso.

---

## 7. Observabilidade e avaliação de agentes {#7-aval}

- **Observabilidade (LangFuse):** capturar o trace completo — cada Thought/Action/
  Observation, latência e tokens por passo. Permite achar ineficiências (ferramenta
  chamada à toa, loops) e estimar custo.
- **Avaliação:** além das métricas de resposta (RAGAS — Faithfulness, ResponseRelevancy),
  agentes pedem olhar a **trajetória**: o agente escolheu as ferramentas certas? quantos
  passos? O *recall de ferramentas* (chamou as necessárias e só elas) é tão importante
  quanto a resposta final.
- **Custo de produção:** estimar "quanto custaria rodar o agente 1.000 vezes" — nº médio
  de passos × tokens por passo × preço do modelo. É o que torna o Adaptive RAG atraente.

---

## 8. Mapa: conceito → componente Haystack {#8-mapa}

| Conceito da aula | Componente Haystack | Observação |
|---|---|---|
| Laço ReAct (Thought/Action/Observation) | `haystack.components.agents.Agent` | tool-calling nativo, `max_agent_steps`, `exit_conditions` |
| Ferramenta | `haystack.tools.Tool` / `ComponentTool` | nome + descrição + JSON schema + função |
| Execução de ferramenta | `ToolInvoker` (interno ao `Agent`) | valida schema, executa, devolve observação |
| LLM com tool calling | `OpenAIChatGenerator` (→ Groq) | `llama-3.3-70b-versatile` suporta tools |
| Classificador + 3 caminhos (Adaptive) | LLM + `ConditionalRouter` | mesmo padrão do CRAG (Aula 8) |
| Web search | `TavilyWebSearch` | com *fallback* offline |
| Dados estruturados | ferramenta *text-to-SQL* sobre SQLite | guard: só `SELECT` |
| Observabilidade | `LangfuseConnector` | trace do agente e dos passos |
| Avaliação | RAGAS / DeepEval | + métricas de trajetória |

> **Por que não LangGraph?** O roteiro usa LangGraph para o laço do agente. O Haystack
> `Agent` entrega o mesmo laço ReAct (com guards) de forma nativa, mantendo um stack só.
> Se um dia a turma quiser ver LangGraph, vale como comparação — mas não é necessário
> para cumprir os objetivos desta aula.

---

## 9. Aplicações em Direito e Segurança Pública {#9-aplicacoes}

- **Assistente investigativo:** combina acervo de jurisprudência, banco de ocorrências e
  fatos atuais — o agente decide as fontes conforme a pergunta.
- **Triagem de demandas:** o Adaptive RAG separa perguntas triviais (resposta direta) das
  que exigem pesquisa profunda, controlando custo em escala.
- **Análise comparativa de jurisprudência:** perguntas multi-hop ("compare X e Y, aponte
  divergências") que só o caminho multi-step resolve bem.
- **Rastreabilidade:** trace por passo (ferramenta + fonte) atende à exigência de
  fundamentação verificável em pareceres e laudos.

---

## 10. Referências {#10-referencias}

JEONG, Soyeong et al. **Adaptive-RAG: Learning to Adapt Retrieval-Augmented Large Language
Models through Question Complexity**. In: *Findings of the ACL 2024*. Disponível em:
https://arxiv.org/abs/2403.14403. Acesso em: 21 abr. 2026.

YAO, Shunyu et al. **ReAct: Synergizing Reasoning and Acting in Language Models**. In:
*Proceedings of ICLR 2023*. Disponível em: https://arxiv.org/abs/2210.03629. Acesso em:
21 abr. 2026.

DEEPSET. **Haystack Agents and Tools Documentation**. Disponível em:
https://docs.haystack.deepset.ai/docs/agent. Acesso em: 17 jun. 2026.

LANGCHAIN-AI. **LangGraph: Build Stateful, Multi-Actor Applications with LLMs**.
Disponível em: https://langchain-ai.github.io/langgraph. Acesso em: 21 abr. 2026.

---

*MBA em RAG & CAG Aplicados a Direito e Segurança Pública · Aula 10 de 12*
