# Guia — LightRAG Server e WebUI (visualizar o grafo) · Aula 9

Este guia mostra como subir o **LightRAG Server** com a **WebUI** para o aluno
**ver o grafo de conhecimento visualmente**, indexar documentos e fazer consultas
pelo navegador. Configuração desta aula:

- **LLM:** Groq (`llama-3.3-70b-versatile`) — API OpenAI-compatible
- **Embedding:** Ollama **bge-m3** (1024 dim) — local
- **Rerank:** **Jina** (API de nuvem) — opcional, melhora `hybrid/mix`
- **Storage:** nativo em arquivo (NetworkX + NanoVectorDB + JSON) — sem banco

> **Importante:** a Aula 9 foi padronizada em **bge-m3**. Os scripts (`01_indexar_grafo.py`
> …) e o servidor usam o **mesmo modelo e dimensão**, então a WebUI abre exatamente o
> grafo que você indexou via script. Se você já tinha indexado com `nomic-embed-text`,
> rode `python 01_indexar_grafo.py --recriar` para reconstruir em bge-m3.

---

## 1. Pré-requisitos

```bash
# 1) Pacote do servidor (inclui a WebUI e a API)
pip install "lightrag-hku[api]"

# 2) Modelo de embedding local
ollama pull bge-m3
```

Você vai precisar de:
- **Chave da Groq** (LLM).
- **Chave da Jina** (rerank) — opcional; dá para rodar sem rerank (veja §5).

---

## 2. Configurar o `.env`

O `lightrag-server` lê o `.env` **da pasta de onde é iniciado**. Para compartilhar o
mesmo grafo dos scripts, vamos iniciar o servidor de dentro de `aula9/`.

1. Copie o modelo para `aula9/.env`:

   ```bash
   # a partir de aula9/
   copy server\env.example.server .env        # Windows
   # cp server/env.example.server .env         # Linux/macOS
   ```

2. Edite `aula9/.env` e preencha:
   - `LLM_BINDING_API_KEY` → sua chave **Groq** (mesmo valor do `GROQ_API_KEY`).
   - `RERANK_BINDING_API_KEY` → sua chave **Jina** (ou desligue o rerank, §5).

> O `.env` do servidor fica em `aula9/.env` e **não conflita** com o `.env` do projeto
> (na raiz), que os scripts continuam usando.

---

## 3. Subir o servidor

A partir da pasta `aula9/` (assim `WORKING_DIR=./rag_storage` aponta para o grafo dos scripts):

```bash
cd aula9\server
lightrag-server
```

- Abra **http://localhost:9621** no navegador.
- (Windows) use o modo padrão `lightrag-server`. O modo `lightrag-gunicorn` **não roda no Windows**.
- Sempre que mudar o `.env`, **feche e reabra o terminal** antes de reiniciar.

---

## 4. Usar a WebUI

Na interface você tem três áreas principais:

1. **Documents** — subir/indexar novos documentos (ou ver os já indexados). Se você já
   rodou `01_indexar_grafo.py`, o grafo aparece aqui sem precisar reindexar. Para indexar
   pela WebUI, suba o `datasets/corpus_juridico.txt`.
2. **Knowledge Graph** — a **visualização interativa do grafo**: nós (entidades) e
   arestas (relações), com layout por gravidade, busca de nós, filtro de subgrafo e zoom.
   É a parte central desta aula — o aluno vê as entidades (Lei 12.850, MPF, STF,
   Colaboração Premiada, operações) e como se conectam.
3. **Retrieval** — uma interface de consulta com os modos `naive / local / global /
   hybrid / mix`, para comparar as respostas (o mesmo que o script `02_consultar_modos.py`,
   só que visual).

---

## 5. Rerank — ligar, trocar ou desligar

O **Ollama não faz rerank** (não tem endpoint de rerank). Por isso o rerank sai por uma
API externa ou por um serviço local dedicado:

- **Jina (padrão deste guia, nuvem):** já configurado no `.env`. Só preencher a chave.
- **Cohere (nuvem):** descomente o bloco Cohere no `.env` e comente o da Jina.
- **Local (avançado):** suba um reranker `bge-reranker-v2-m3` via **vLLM/TEI** expondo um
  endpoint compatível com Cohere e use:
  ```
  RERANK_BINDING=cohere
  RERANK_MODEL=BAAI/bge-reranker-v2-m3
  RERANK_BINDING_HOST=http://localhost:8000/rerank
  RERANK_BINDING_API_KEY=qualquer-coisa
  ```
  (normalmente exige GPU).
- **Desligar:** `RERANK_BINDING=null`. O servidor funciona normalmente, sem o *boost* de
  rerank. O rerank é só qualidade de query — **não exige reindexar** para ligar/desligar.

---

## 6. Fluxo recomendado para a aula

```bash
# 1) indexar o grafo via script (bge-m3) — ou pular e indexar pela WebUI
cd aula9/scripts
python 01_indexar_grafo.py --recriar

# 2) subir o servidor a partir de aula9/ e visualizar
cd ..
copy server\env.example.server .env   # preencha as chaves
lightrag-server
# abrir http://localhost:9621 -> aba Knowledge Graph
```

---

## 7. Problemas comuns

- **Grafo vazio na WebUI:** você iniciou o servidor de uma pasta diferente de `aula9/`
  (então `WORKING_DIR` não aponta para o grafo). Inicie de `aula9/` ou defina
  `WORKING_DIR` com caminho absoluto no `.env`.
- **Erro de dimensão de embedding:** o grafo foi indexado com outro modelo (ex.: nomic,
  768). Reindexe com bge-m3 (`01_indexar_grafo.py --recriar`) — embedding e dimensão
  precisam ser os mesmos da indexação.
- **`json: unsupported value: NaN` (erro 500) ao indexar com bge-m3:** bug conhecido do
  Ollama (o `bge-m3` gera NaN em alguns embeddings, ligado ao flash attention). Corrija
  desligando o flash attention e reiniciando o Ollama: feche o Ollama, rode
  `setx OLLAMA_FLASH_ATTENTION 0` (Windows) e reabra; depois `01_indexar_grafo.py --recriar`.
  Se persistir, atualize o Ollama (+ `ollama pull bge-m3`) ou troque para `nomic-embed-text`.
- **Rerank dá erro 401/403:** chave da Jina/Cohere ausente ou inválida — preencha
  `RERANK_BINDING_API_KEY` ou use `RERANK_BINDING=null`.
- **LLM lento/timeout:** confirme a chave Groq e o modelo `llama-3.3-70b-versatile`
  (não use modelos de raciocínio na indexação).
