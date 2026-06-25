# Scripts da Aula 5 — Guia de Uso

Scripts Python simples, de linha de comando, que reproduzem os assuntos práticos
da Aula 5 (**Avaliação e Observabilidade: RAGAS, DeepEval e LangFuse**):

**RAGAS** (4 métricas) · **DeepEval** (testes de qualidade) · **LangFuse Scores API** · juiz **Groq** · embeddings **Ollama**

> **Coerência corpus × perguntas:** para a avaliação fazer sentido, as perguntas
> precisam ter resposta no índice avaliado. Por isso o `01` **gera as perguntas a
> partir dos próprios acórdãos do TCU** (índice `aula4_hibrido` da Aula 4). Assim a
> resposta sempre existe no corpus e fica fácil de explicar/visualizar.

---

## 1. Pré-requisitos

1. **Python 3.10+** com o ambiente virtual do curso ativado.
2. **OpenSearch** com o índice **`aula4_hibrido`** já criado — rode o
   `01_indexar_hibrido.py` da **Aula 4** antes.
3. **Ollama** com o modelo de embedding: `ollama pull nomic-embed-text`.
4. **Chave da Groq** no `.env` (gera perguntas e é o juiz da avaliação).
5. **LangFuse** (para o script 05) — veja `GUIA_LANGFUSE_WINDOWS.md` da **Aula 3**.

---

## 2. Instalação das dependências

Com o ambiente virtual **ativado**, dentro da pasta `scripts`:

```bash
pip install -r requirements.txt
```

> **Atenção (RAGAS):** o RAGAS importa `langchain_community.chat_models.vertexai`,
> que existe na versão **0.4.1** mas foi removido na 0.4.2. O `requirements.txt` já
> fixa `langchain-community==0.4.1`.

---

## 3. Ordem recomendada e como usar cada script

Rode tudo de dentro da pasta `aula5/scripts`. **A ordem importa.**

### `00_check_ambiente.py` — confere se está tudo pronto
```bash
python 00_check_ambiente.py
python 00_check_ambiente.py --testar-groq
```

### `01_gerar_perguntas.py` — cria perguntas a partir dos acórdãos
Para cada acórdão do TCU, o LLM gera 1 pergunta + a resposta (ground_truth).
Salva `perguntas_geradas.json`.
```bash
python 01_gerar_perguntas.py --limite 10
```

### `02_gerar_dataset.py` — produz contexts + answer (roda o RAG)
Roda o RAG híbrido sobre o índice e salva `dataset_avaliacao_completo.json`.
Usa automaticamente as perguntas geradas pelo `01`.
```bash
python 02_gerar_dataset.py --indice aula4_hibrido
```

### `03_ragas_avaliar.py` — 4 métricas RAGAS (juiz Groq)
```bash
python 03_ragas_avaliar.py
python 03_ragas_avaliar.py --limite 10
```

### `04_deepeval_testes.py` — testes de qualidade (PASS/FAIL)
Faithfulness, AnswerRelevancy, Hallucination, Toxicity, Bias.
```bash
python 04_deepeval_testes.py
python 04_deepeval_testes.py --limite 3
```

### `05_langfuse_scores.py` — envia os scores ao LangFuse
Lê os resultados do `03` e registra como scores (Scores API).
```bash
python 05_langfuse_scores.py
```

### `06_chat_rag_langfuse.py` — chat RAG instrumentado (LLM-as-a-Judge nativo)
Chat interativo cujas perguntas/respostas viram traces no LangFuse. Depois, na UI
do LangFuse, você configura um **avaliador LLM-as-a-Judge (Ragas)** que pontua os
traces **automaticamente** — sem RAGAS em código. Passo a passo em
`GUIA_LANGFUSE_LLM_AS_JUDGE.md`.
```bash
python 06_chat_rag_langfuse.py --indice aula4_hibrido
```
Diferença para o `05`: no `05` o seu código calcula e envia os scores (RAGAS);
no `06` quem avalia é o **próprio LangFuse**, continuamente, sobre os traces.

---

## 4. Resumo de dependências por script

| Script | OpenSearch | Ollama | Groq | RAGAS | DeepEval | LangFuse |
|--------|:----------:|:------:|:----:|:-----:|:--------:|:--------:|
| 00_check_ambiente | ✓ (checa) | ✓ (checa) | ✓ (checa) | ✓ (checa) | ✓ (checa) | ✓ (checa) |
| 01_gerar_perguntas | — | — | ✓ | — | — | — |
| 02_gerar_dataset | ✓ | ✓ | ✓ | — | — | — |
| 03_ragas_avaliar | — | ✓ | ✓ | ✓ | — | — |
| 04_deepeval_testes | — | — | ✓ | — | ✓ | — |
| 05_langfuse_scores | — | — | — | — | — | ✓ |
| 06_chat_rag_langfuse | ✓ | ✓ | ✓ | — | — | ✓ |

> `_comum.py` não é executado diretamente: carrega o `.env`, lê as perguntas/acórdãos,
> prepara os juízes LangChain (Groq/Ollama) e as configurações.

---

## 5. Observações

- **Perguntas geradas vs. originais:** por padrão o `02` usa `perguntas_geradas.json`
  (do `01`). Se quiser usar as 50 perguntas originais da aula, passe
  `--perguntas ..\datasets\corpus_avaliacao_aula5.json` — mas note que elas são de
  direito penal e **não** casam com o corpus do TCU (a maioria dará "não consta").
- **Por que gerar perguntas do acórdão?** Garante que a resposta existe no índice,
  então as métricas medem a qualidade do RAG (e não a ausência do assunto no corpus).
- **Juiz = Groq:** avaliação "LLM-as-judge". Para custo/tempo, use `--limite`.
- **LangFuse:** o `05` cria um trace e anexa as médias do RAGAS como scores.
