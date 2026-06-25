# Projeto Final — API de Ingestão Inteligente + RAG (Aula 12)

Uma **API FastAPI** que recebe um documento (PDF, DOCX, XLSX, imagem, TXT), **decide
sozinha** como extrair e como indexar, e expõe consulta RAG. É a síntese do curso
(aulas 2–11): Haystack, Docling, Ollama, OpenSearch, LightRAG e Groq.

## A decisão automática (o "cérebro")

```
Upload  →  [1] probe (sinais baratos: extensão, texto, imagens)
        →  [2] AGENTE Haystack escolhe a ferramenta de extração:
                 planilha → pandas | texto → Docling | escaneado/figura → Docling+OCR
        →  [3a] HEURÍSTICA escolhe o DESTINO:
                 texto/tabela → OpenSearch
                 texto longo e rico em entidades → LightRAG (grafo, multi-hop)
        →  [3b] Se OpenSearch, AVALIADOR escolhe a melhor TÉCNICA DE CHUNKING
                 (fixo | recursivo | sentence-window | semântico | hierárquico | tabela)
        →  [4] /consulta roteia a busca (OpenSearch ou grafo) + gera resposta (Groq)
```

- **Extração**: decidida por um **Agente** (LLM tool-calling, aula 10) a partir dos sinais
  do documento. Cada ferramenta extrai e devolve só um resumo ao agente (o conteúdo grande
  fica num cache, fora do contexto do LLM).
- **Figuras/escaneados**: viram **texto via OCR do Docling** (`do_ocr=True`).
- **Destino de indexação**: heurística transparente (densidade de entidades) escolhe
  **OpenSearch** (padrão) ou **LightRAG** (grafo); dá para forçar via `estrategia=auto|opensearch|grafo`.
- **Estratégia de chunking** (só no OpenSearch): um **avaliador transparente** analisa a
  estrutura do documento e escolhe a melhor técnica — cada uma é um componente **nativo do
  Haystack**:

  | Técnica | Componente Haystack | Escolhida quando |
  |---|---|---|
  | `tabela` | (1 chunk por tabela) | documento tem tabelas |
  | `hierarquico` | `HierarchicalDocumentSplitter` | ≥3 títulos/seções (preserva a estrutura) |
  | `sentenca_janela` | `DocumentSplitter` (sentenças + overlap) | texto de lei / denso em artigos (precisão) |
  | `semantico` | `EmbeddingBasedDocumentSplitter` | texto longo e heterogêneo (corta por tópico) |
  | `recursivo` | `RecursiveDocumentSplitter` | texto corrido (default robusto) |
  | `fixo` | `DocumentSplitter` (palavras) | documento curto / baseline |

  Dá para forçar via `chunking=auto|fixo|recursivo|sentenca_janela|semantico|hierarquico`.
  O relatório de ingestão devolve a técnica escolhida **e o motivo** (didático).
- **Embedding**: Ollama `nomic-embed-text` (o mesmo no OpenSearch, no LightRAG e no
  chunking semântico).

## Estrutura

```
projeto_final/
├── app/
│   ├── main.py        # FastAPI: /ingestao, /consulta, /health, /metrics
│   ├── config.py      # configs (Groq/Ollama/OpenSearch/LightRAG)
│   ├── extracao.py    # probe + AGENTE Haystack + ferramentas de extração (Docling/pandas/OCR)
│   ├── indexacao.py   # decisão de destino + AVALIADOR de chunking (5 técnicas) + escrita
│   ├── consulta.py    # RAG roteado por storage + Groq
│   └── modelos.py     # schemas Pydantic
├── interface.py       # interface Gradio (consome a API)
├── cliente_exemplo.py # exemplos de uso
├── requirements.txt
├── .env.example
└── (datasets/docker-compose.yml — sobe OpenSearch + Dashboards + LangFuse)
```

## Como rodar

```bash
# 1) infraestrutura (OpenSearch etc.)
docker compose -f ../datasets/docker-compose.yml up -d     # ou seu OpenSearch local

# 2) modelos / chaves
ollama pull nomic-embed-text
cp .env.example .env        # preencha GROQ_API_KEY (e ajuste OPENSEARCH_*)

# 3) dependências e API
pip install -r requirements.txt
uvicorn app.main:app --reload        # de dentro de projeto_final/ (terminal 1)

# 4) interface Gradio (opcional, terminal 2)
python interface.py                  # abre em http://localhost:7860

# 5) ou testar sem interface
python cliente_exemplo.py algum_documento.pdf
# ou a doc interativa:  http://localhost:8000/docs
```

A **interface Gradio** (`interface.py`) tem 3 abas: **Ingestão** (upload + relatório da
decisão, com override de destino e de chunking), **Consulta** (pergunta → resposta + fontes)
e **Status** (`/health`). Ela só consome a API — não duplica lógica de RAG.

## Endpoints

| Método | Rota | Descrição |
|---|---|---|
| POST | `/ingestao` | upload de documento → extrai (agente) → indexa (heurística) → **relatório da decisão** |
| POST | `/consulta` | pergunta → resposta RAG + fontes (roteado p/ OpenSearch ou grafo) |
| GET | `/health` | status de OpenSearch, Groq, embedding, LangFuse |
| GET | `/metrics` | contadores (ingestões, consultas, erros, uptime) |

Exemplo de `/ingestao` (curl) — `estrategia` força o destino, `chunking` força a técnica:

```bash
curl -F "arquivo=@laudo.pdf" "http://localhost:8000/ingestao?estrategia=auto&chunking=auto"
```

Resposta (resumida) — mostra **a decisão**, para o aluno entender o raciocínio:

```json
{
  "ok": true,
  "relatorio": {
    "arquivo": "laudo.pdf",
    "complexidade": "complexo",
    "tecnica_extracao": "ocr",
    "estrutura": {"extensao": ".pdf", "n_paginas": 8, "tem_imagens": true, "provavel_escaneado": true},
    "destino": "opensearch",
    "motivo_destino": "texto/tabela direto (12 entidades distintas < 30)",
    "chunking": "hierarquico",
    "motivo_chunking": "documento estruturado em secoes (5 titulos) -> hierarquico",
    "n_chunks": 23
  }
}
```

## Observações didáticas

- **Por que agente para extração e heurística para indexação/chunking?** A escolha da
  técnica de extração é ambígua (depende de ler os sinais) → o agente brilha. O destino e a
  técnica de chunking são regras claras e auditáveis (estrutura do documento) → avaliador
  heurístico transparente é melhor (e sem custo de LLM). Para virar decisão por LLM, basta
  trocar `avaliar_chunking` por uma chamada ao agente.
- **Avaliador de chunking**: ajuste os limiares (`n_titulos`, `n_artigos`, nº de palavras) em
  `avaliar_chunking` (`indexacao.py`). O `chunking=...` no `/ingestao` força qualquer técnica
  — útil para o aluno comparar o efeito de cada uma sobre a mesma fonte.
- **LightRAG (grafo)** é escolhido só quando o texto é longo e rico em entidades (bom para
  perguntas multi-hop). Ajuste `LIMIAR_ENTIDADES` / `MIN_PALAVRAS_GRAFO` em `indexacao.py`.
- **LLM**: use `llama-3.3-70b-versatile` (tool calling). **Não use gpt-oss** (reasoning):
  o tool-calling do agente fica instável.
- **Pesos**: Docling baixa modelos de layout/OCR na 1ª execução; o LightRAG faz várias
  chamadas de LLM ao construir o grafo (rota mais cara — por isso é seletiva).
- **Produção**: o `docker-compose` traz OpenSearch + Dashboards + LangFuse; em produção,
  rode tudo on-premise (dados sensíveis) e ative auth/SSL.
