# Scripts da Aula 9 — Guia de Uso

Scripts Python simples, de linha de comando, sobre **Graph RAG** (RAG com grafos de
conhecimento) usando o **LightRAG**.

Stack: **LightRAG** · **Groq** (LLM) · **Ollama** (embeddings) · **storage nativo em
arquivo** (NetworkX + NanoVectorDB + JSON).

> **Decisões desta aula:**
> - **Sem banco de grafo e sem OpenSearch.** O LightRAG guarda tudo em arquivos no
>   `working_dir` (`aula9/rag_storage/`): o grafo em `graph_chunk_entity_relation.graphml`
>   (NetworkX), os vetores em `vdb_*.json` (NanoVectorDB) e os textos/cache em `kv_store_*.json`.
>   É o que a própria documentação do LightRAG recomenda para test/dev — zero instalação extra.
> - **LLM = Groq `llama-3.3-70b-versatile`** (instruct). O LightRAG **desaconselha modelos de
>   raciocínio** (gpt-oss etc.) na extração de entidades. Para trocar: `AULA9_LLM_MODEL` no `.env`.
> - **Embeddings = Ollama `bge-m3`** (1024 dimensões), local — o **mesmo** modelo do
>   LightRAG Server/WebUI, para que a interface visualize o grafo construído pelos scripts.
> - **Corpus = `datasets/corpus_juridico.txt`** (crime organizado, colaboração premiada,
>   STF/MPF, operações) — feito para multi-hop, ideal para demonstrar o ganho do grafo.

---

## 1. Por que Graph RAG?

O RAG convencional recupera *chunks* de texto isolados — ótimo para perguntas factuais
diretas, fraco para **raciocínio multi-hop** (conectar várias entidades). O Graph RAG
extrai um **grafo de conhecimento** (entidades + relações) do corpus e navega por ele,
respondendo perguntas como "quais as conexões entre o MPF e a colaboração premiada?".

O LightRAG monta esse grafo automaticamente: *chunking* → o LLM **extrai entidades e
relações** de cada trecho → grafo (NetworkX) + índices vetoriais. Depois, consulta em
quatro modos:

| Modo | O que faz | Quando usar |
|------|-----------|-------------|
| `naive` | busca vetorial simples (sem grafo) | baseline / comparação |
| `local` | entidades + vizinhos no grafo | perguntas sobre pessoas/leis/casos |
| `global` | sumários de comunidades | perguntas amplas/temáticas |
| `hybrid` | local + global + chunks | produção / melhor qualidade |

---

## 2. Pré-requisitos

1. **Python 3.10+** com o ambiente virtual do curso ativado.
2. **Ollama** com o modelo de embedding: `ollama pull bge-m3`.
3. **Chave da Groq** no `.env` (`GROQ_API_KEY`).
4. Instalar as dependências:

```bash
pip install -r requirements.txt
```

> Não precisa de OpenSearch, Docker, Neo4j nem GPU para esta aula.

---

## 3. Ordem recomendada e como usar cada script

Rode tudo de dentro da pasta `aula9/scripts`.

### `00_check_ambiente.py` — confere se está tudo pronto
```bash
python 00_check_ambiente.py --testar-groq
```

### `01_indexar_grafo.py` — constrói o grafo de conhecimento
Lê o corpus e extrai entidades/relações via LLM (etapa "cara", roda uma vez e fica salva).
```bash
python 01_indexar_grafo.py
python 01_indexar_grafo.py --recriar   # apaga o working_dir e reconstrói
```

### `02_consultar_modos.py` — compara os 4 modos de busca
Mesma pergunta em `naive / local / global / hybrid`, lado a lado.
```bash
python 02_consultar_modos.py --pergunta "o que e colaboracao premiada e quem pode firmar?"
python 02_consultar_modos.py --pergunta "..." --modos naive,hybrid
```

### `03_investigacao.py` — perguntas multi-hop (onde o grafo brilha)
Conexões entre entidades, redes, panorama de operações. Usa `hybrid` por padrão.
```bash
python 03_investigacao.py                 # roda perguntas de exemplo
python 03_investigacao.py --pergunta "quais as conexoes entre o MPF e a colaboracao premiada?"
python 03_investigacao.py --modo local
```

### `04_explorar_grafo.py` — inspeciona o grafo gerado
Mostra nº de entidades/relações, os "hubs" (mais conectados), vizinhos de uma entidade e
exporta um subgrafo. Não usa Groq/Ollama (lê só o arquivo do grafo).
```bash
python 04_explorar_grafo.py
python 04_explorar_grafo.py --entidade "Colaboracao Premiada"
python 04_explorar_grafo.py --entidade "MPF" --exportar subgrafo.json
```

### `05_visualizar_grafo.py` — visualização interativa (HTML) para apresentar
Gera um **HTML autossuficiente** (vis-network) a partir do GraphML: nós coloridos por
tipo de entidade, tamanho por grau (hubs maiores), descrição da relação no hover, com
zoom/arraste/física. Abra no navegador para apresentar em sala (não precisa de servidor).
```bash
python 05_visualizar_grafo.py                          # gera aula9/grafo_juridico.html
python 05_visualizar_grafo.py --entidade "Criptomoedas"   # foca uma entidade + vizinhos
python 05_visualizar_grafo.py --saida grafo.html
```
> Precisa de internet ao abrir (carrega o vis-network via CDN). Alternativa totalmente
> interativa: a aba **Knowledge Graph** do LightRAG Server (veja `../server/`).

---

## 4. Resumo de dependências por script

| Script | LightRAG | Groq | Ollama | NetworkX |
|--------|:--------:|:----:|:------:|:--------:|
| 00_check_ambiente | ✓ (checa) | ✓ (checa) | ✓ (checa) | — |
| 01_indexar_grafo | ✓ | ✓ | ✓ | — |
| 02_consultar_modos | ✓ | ✓ | ✓ | — |
| 03_investigacao | ✓ | ✓ | ✓ | — |
| 04_explorar_grafo | — | — | — | ✓ |

> `_comum.py` não é executado diretamente: carrega o `.env`, define o LLM (Groq) e os
> embeddings (Ollama) do LightRAG e cria a instância com storage em arquivo (`criar_rag`).

---

## 5. Visualizar o grafo: LightRAG Server + WebUI

Para o aluno **ver o grafo visualmente** (nós, arestas, comunidades) e consultar pelo
navegador, use o **LightRAG Server**. Como a aula está padronizada em **bge-m3**, a WebUI
abre o **mesmo grafo** que os scripts constroem.

Resumo (passo a passo completo em [`../server/GUIA_SERVER_WEBUI.md`](../server/GUIA_SERVER_WEBUI.md)):

```bash
pip install "lightrag-hku[api]"
ollama pull bge-m3
# copie aula9/server/env.example.server para aula9/.env e preencha as chaves
cd aula9
lightrag-server          # abre em http://localhost:9621
```

Config: **LLM = Groq** (`llama-3.3-70b-versatile`), **Embedding = Ollama bge-m3**,
**Rerank = Jina (API)**. Inicie o servidor de dentro de `aula9/` para que ele use o mesmo
`./rag_storage` dos scripts. Na aba **Knowledge Graph** está a visualização interativa.

> O Ollama não faz rerank — por isso o rerank usa a API da Jina (ou Cohere). Dá para
> desligar com `RERANK_BINDING=null`. Detalhes no guia.

---

## 6. Observações

- **Os acórdãos do TCU não são ideais aqui.** São 1100 casos isolados, com pouca
  sobreposição de entidades entre documentos — o valor do grafo (multi-hop) quase não
  aparece. Por isso usamos o `corpus_juridico.txt`, com entidades que se repetem e se
  conectam. Em produção, o Graph RAG rende mais em acervos com entidades recorrentes
  (redes criminosas, jurisprudência interligada, fluxo financeiro).
- **Custo de indexação.** A construção do grafo faz muitas chamadas de LLM (uma por
  chunk, para extração). É a etapa cara do Graph RAG — por isso fica persistida.
- **Trocar de embedding exige reconstruir.** O modelo de embedding deve ser o mesmo na
  indexação e na consulta; se trocar, rode `01_indexar_grafo.py --recriar`.
- **Escalar depois:** quando o acervo crescer, dá para mover o storage para Neo4j (grafo)
  ou OpenSearch 3.3+ (vetores) só ajustando variáveis de ambiente do LightRAG — sem
  mudar a lógica dos scripts. Nesta aula ficamos no nativo em arquivo por simplicidade.

---

## 7. Problema conhecido: Ollama + bge-m3 retorna NaN (erro 500)

Ao indexar (`01_indexar_grafo.py`) você pode ver:

```
ERROR: Error in ollama_embed: failed to encode response: json: unsupported value: NaN (status code: 500)
```

É um **bug conhecido do Ollama com o `bge-m3`** (não é dos scripts): o modelo gera
valores **NaN** em alguns embeddings e o Ollama não consegue serializar o JSON. A causa
está ligada ao **flash attention**.

**Correção (mantém o bge-m3):**

1. Feche o Ollama (bandeja → Quit).
2. PowerShell: `setx OLLAMA_FLASH_ATTENTION 0`
3. Reabra o Ollama (para subir com a variável).
4. Reindexe: `python 01_indexar_grafo.py --recriar`

(Se você roda o servidor à mão: `$env:OLLAMA_FLASH_ATTENTION="0"; ollama serve`.)

**Se persistir:** atualize o Ollama (versões recentes tratam NaN) e
`ollama pull bge-m3` de novo; ou, como último recurso, troque o embedding para
`nomic-embed-text` (`AULA9_EMBED_MODEL=nomic-embed-text` no `.env` + `--recriar`) — mas
aí use o mesmo modelo no LightRAG Server (nomic/768) para manter o alinhamento.
