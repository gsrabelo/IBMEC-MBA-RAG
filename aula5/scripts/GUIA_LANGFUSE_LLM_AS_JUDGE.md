# Guia — LLM-as-a-Judge (Ragas) no LangFuse

Este guia mostra como usar o recurso **nativo** de avaliação do LangFuse: os
**avaliadores LLM-as-a-Judge** (templates mantidos em parceria com a **Ragas**).
Diferente do script `05_langfuse_scores.py` (que *envia* notas calculadas em código),
aqui o **próprio LangFuse** pontua automaticamente cada resposta do seu RAG.

Fluxo da demonstração:
1. Rode o chat `06_chat_rag_langfuse.py` e faça algumas perguntas → cada turno vira um **trace** no LangFuse.
2. Configure um **avaliador LLM-as-a-Judge** na interface, filtrando por esses traces.
3. O LangFuse avalia sozinho (faithfulness/hallucination, relevância, etc.) e mostra os scores.

> Referência oficial: https://langfuse.com/docs/evaluation/evaluation-methods/llm-as-a-judge

---

## 0. Pré-requisitos

- **LangFuse rodando** (self-hosted via Podman — veja `GUIA_LANGFUSE_WINDOWS.md` da Aula 3)
  com as chaves no `.env` (`LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_BASE_URL`).
- Uma **chave de LLM** para o juiz (pode ser a própria Groq).
- Ter gerado alguns traces:
  ```
  python 06_chat_rag_langfuse.py --indice aula4_hibrido
  ```
  Faça 3–5 perguntas (ex.: sobre os acórdãos) e saia com `sair`. Cada pergunta
  cria um trace chamado **`chat-rag-aula5`**.

---

## 1. Conectar um modelo de LLM (juiz) no LangFuse

O avaliador precisa de um LLM que faça o julgamento.

1. No LangFuse: **Settings → LLM Connections → + Add connection**.
2. Como a Groq é compatível com a API da OpenAI, escolha o provedor **OpenAI** e informe:
   - **API Key:** sua `GROQ_API_KEY`
   - **Base URL:** `https://api.groq.com/openai/v1`
   - **Model:** um modelo capaz, ex. `llama-3.3-70b-versatile`
3. Salve.

> Importante: o modelo-juiz precisa **suportar saída estruturada (JSON)** — por isso
> prefira um modelo maior (70B) em vez do 8B. Modelos pequenos erram o formato e o
> avaliador falha.

---

## 2. Criar o avaliador (template Ragas)

1. Vá em **Evaluation → LLM-as-a-Judge → + Set up Evaluator**.
2. **Default model:** selecione a conexão criada no passo 1.
3. **Pick an Evaluator → Managed**: escolha um template da **Ragas**. Para RAG, os mais úteis:
   - **Hallucination** (≈ faithfulness: a resposta está ancorada no contexto?)
   - **Context Relevance** (os trechos recuperados são relevantes para a pergunta?)
   - **Answer Relevance / Helpfulness** (a resposta responde à pergunta?)

   Comece com **Hallucination** (depois repita para os outros).

---

## 3. Escolher quais dados avaliar

1. Em **Choose which Data to Evaluate**, selecione **Live Traces** (ou *Observations*,
   se sua versão recomendar — para o nosso chat, *Traces* funciona bem).
2. **Filtro:** filtre por **trace name = `chat-rag-aula5`** (o nome que o script usa).
   Assim o avaliador roda só nos traces do nosso chat.
3. **Run on:** novos traces (pode marcar para incluir os já existentes / backfill).
4. **Sampling:** 100% para a demonstração (em produção, use 5–10% para controlar custo).
5. Use o **Preview** (últimas 24h) para confirmar que há traces batendo no filtro.

---

## 4. Mapear as variáveis do prompt do avaliador

O template usa variáveis como `{{input}}`, `{{output}}` e (nos de RAG) `{{context}}`.
Você precisa dizer ao LangFuse **de onde** tirar cada uma no trace:

- `{{input}}`  → a **pergunta** (input do trace / da raiz do pipeline)
- `{{output}}` → a **resposta** do LLM (output do trace)
- `{{context}}` / `{{retrieved_context}}` → os **trechos recuperados** (saída do
  passo de retrieval/fusão — no nosso pipeline, o componente `juntar`)

Dicas:
- Use o **preview ao vivo**: o LangFuse mostra o prompt já preenchido com dados reais
  dos últimos traces — ajuste o mapeamento até o preview ficar correto.
- Se o dado estiver **aninhado** em um JSON, use **JSONPath** (ex.: `$.replies[0]`
  para a resposta, ou um caminho até a lista de documentos do contexto).

---

## 5. Disparar e ver os resultados

1. Rode o chat de novo e faça perguntas:
   ```
   python 06_chat_rag_langfuse.py --indice aula4_hibrido
   ```
2. Em segundos/minutos, o avaliador roda nos novos traces.
3. Veja os scores:
   - **Tracing → Traces:** colunas de score por trace (Hallucination, Context Relevance…).
   - **Dashboards:** médias e evolução ao longo do tempo (avaliação contínua).

---

## 6. Depurar o avaliador

- Toda execução do juiz gera um trace próprio. Para inspecionar, filtre a tabela de
  traces pelo *environment* **`langfuse-llm-as-a-judge`**.
- Status possíveis: **Completed**, **Error** (clique para ver o motivo), **Delayed**
  (rate limit, re-tentando), **Pending** (na fila).
- Erros comuns: modelo-juiz sem saída estruturada (use um 70B); mapeamento de
  variáveis errado (revise o passo 4); LLM Connection sem créditos/quota.

---

## 7. Quando usar isto vs. o script `05`

| Abordagem | Quem calcula | Quando usar |
|---|---|---|
| `05_langfuse_scores.py` (RAGAS em código) | seu script (RAGAS) → envia score | avaliação em lote/offline, no seu controle |
| **LLM-as-a-Judge nativo (este guia)** | o **LangFuse** automaticamente | monitoramento contínuo de produção, sem código extra |

Em produção, o padrão é: **Experiments** no desenvolvimento e **LLM-as-a-Judge
automático** sobre os traces reais em produção.

---

*MBA RAG & CAG Aplicados a Direito e Segurança Pública — Aula 5*
