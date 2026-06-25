# Scripts da Aula 11 — Guia de Uso

Scripts Python simples, de linha de comando, sobre **técnicas complementares** ao RAG:
**Time-Aware**, **Compressão (LLMLingua)**, **ColBERT**, **Multimodal (CLIP)** e **DSPy**.

Base do stack: **OpenSearch** (busca densa) · **Ollama** (embeddings) · **Groq** (LLM).
Cada técnica usa, além disso, a ferramenta de referência da sua área. Teoria completa em
[`../teoria/AULA11_TEORIA.md`](../teoria/AULA11_TEORIA.md).

> **Importante:** cada técnica tem dependências próprias (algumas pesadas: torch/faiss/
> CLIP). Instale **só o que for usar** — veja os blocos do `requirements.txt`. O `00_check`
> mostra o que está instalado por técnica.

---

## 1. Pré-requisitos

1. **Python 3.10+** com o ambiente virtual do curso ativado.
2. **Ollama** com o embedding: `ollama pull nomic-embed-text`.
3. **Chave da Groq** no `.env` (`GROQ_API_KEY`); LLM `llama-3.3-70b-versatile`.
4. **OpenSearch** em `localhost:9200`.
5. Dependências por técnica:

```bash
pip install -r requirements.txt          # tudo
# ou só a base + a técnica que for usar (ver requirements.txt)
```

---

## 2. Ordem recomendada e como usar cada script

Rode tudo de dentro da pasta `aula11/scripts`.

### `00_check_ambiente.py` — confere base + libs por técnica
```bash
python 00_check_ambiente.py --testar-groq
```

### `01_indexar.py` — indexa o corpus no OpenSearch (base)
Indexa os 30 documentos (com `data`/`vigente`/`tipo`) + embeddings.
```bash
python 01_indexar.py
python 01_indexar.py --recriar
```

### `02_time_aware.py` — relevância × recência (decay temporal) · leve
Recupera por similaridade e re-ranqueia por `relevância × decay(idade)`. Mostra o ranking
sem e com decay e o filtro de vigência. **Não precisa de lib extra.**
```bash
python 02_time_aware.py --pergunta "regras de licitacao"
python 02_time_aware.py --pergunta "..." --scale 3650 --offset 365 --so-vigentes
```
> O corpus tem leis de 1940–2022, então o `--scale` (meia-vida) é em **anos** (padrão
> 3650 = 10 anos). Para normas operacionais, use um `--scale` menor (decay mais agressivo).

### `03_compressao_llmlingua.py` — compressão de contexto (LLMLingua-2) · médio
Recupera o contexto, comprime com o LLMLingua-2 e responde. Mostra tokens antes/depois.
```bash
pip install llmlingua
python 03_compressao_llmlingua.py --pergunta "o que caracteriza o crime de roubo?" --taxa 0.5
```

### `04_colbert_ragatouille.py` — ColBERT vs busca densa · médio
ColBERT (late interaction) via **PyLate** (PyTorch puro) comparado com a busca densa.
Usa `pylate.rank.rerank` (sem índice PLAID, pois o corpus é pequeno) — **roda em
Windows/CPU sem compilador C++**, ao contrário do ragatouille/colbert clássico (que exige
o MSVC `cl.exe`).
```bash
pip install pylate
python 04_colbert_ragatouille.py --pergunta "prazo de recurso de apelacao"
# ColBERT moderno (melhor em CPU): AULA11_COLBERT_MODEL=lightonai/GTE-ModernColBERT-v1
```

### `05_multimodal_clip.py` — OCR + CLIP + ColPali sobre PDFs · pesado
Pipeline real sobre **PDFs** em `datasets/pdfs/`: renderiza **cada página** como imagem
(PyMuPDF), extrai o texto por página via **Docling com OCR** (`conteudo`) e indexa no
**OpenSearch** em dois índices — **visual** (imagem via CLIP, 512d) e **texto** (OCR via
Ollama, 768d). A busca tem **4 modos** (`--modo`):

- `visual` — CLIP na imagem (bom p/ achar **figuras** por descrição visual);
- `texto` — OCR + embedder de texto (bom p/ o **conteúdo textual** da página);
- `hibrido` — funde visual + texto por RRF (padrão);
- `colpali` — **ColPali** (via `byaldi`, late interaction multimodal) — estado da arte p/
  recuperar **página de documento por texto**.

> Por que os 4: o CLIP "puro" é fraco em página cheia de texto (vê categorias visuais, não
> lê o texto); o modo texto cobre a semântica textual; o ColPali é o mais forte p/ páginas
> de documento. Use `visual` para figuras, `texto`/`hibrido` para conteúdo, `colpali` para
> o melhor de recuperação visual de documentos.

```bash
pip install sentence-transformers pillow pymupdf docling   # visual/texto/hibrido
pip install byaldi                                          # colpali
python 05_multimodal_clip.py --indexar                       # OCR + CLIP + texto no OpenSearch
python 05_multimodal_clip.py --consulta "mapa de calor de criminalidade" --modo hibrido
python 05_multimodal_clip.py --indexar --modo colpali        # indexa com ColPali
python 05_multimodal_clip.py --consulta "..." --modo colpali
```
Sem PDFs na pasta, gera um de exemplo (4 páginas).

### `06_dspy_otimizacao.py` — otimização automática de prompt (DSPy) · médio
Declara um RAG com `ChainOfThought`, compila com `BootstrapFewShot` (usando a Groq) e
compara a resposta antes/depois.
```bash
pip install dspy-ai
python 06_dspy_otimizacao.py --n-treino 5
```

---

## 3. Resumo: peso e dependência por script

| Script | Técnica | Peso | Lib extra |
|--------|---------|:----:|-----------|
| 02_time_aware | Time-Aware | leve | — (OpenSearch puro) |
| 03_compressao_llmlingua | Compressão | médio | llmlingua |
| 06_dspy_otimizacao | DSPy | médio | dspy-ai |
| 04_colbert_ragatouille | ColBERT | médio | pylate (torch; sem compilador) |
| 05_multimodal_clip | Multimodal | pesado | sentence-transformers + pillow |

> `_comum.py` não é executado: carrega o `.env`, o corpus benchmark, e os blocos comuns
> (OpenSearch, Ollama, Groq, busca densa).

---

## 4. Observações

- **Time-Aware é o mais alinhado ao stack** e o mais barato — provavelmente o de maior uso
  prático no Direito (vigência/recência). No OpenSearch nativo, o equivalente é o
  `function_score` com `exp` no campo de data; aqui fazemos o decay no Python para ficar
  didático e robusto a mapeamento de data.
- **As técnicas pesadas baixam modelos grandes na 1ª execução** (ColBERTv2 ~440MB, CLIP
  ~600MB, LLMLingua-2 ~560MB). Rode com internet e espaço em disco.
- **ColBERT no Windows/CPU:** usamos **PyLate** (PyTorch puro) justamente para evitar o
  erro do `ragatouille`/`colbert` clássico, que compila uma extensão C++ em runtime e
  exige o Visual Studio Build Tools (`cl.exe`). PyLate não precisa de compilador.
- **ColPali (modo `--modo colpali` do 05) precisa de GPU na prática:** é um VLM de ~3B; em
  CPU é muito lento e o `flash-attn` não instala no Windows sem CUDA. Sem GPU, rode o
  ColPali no Google Colab ou use os modos `visual`/`texto`/`hibrido`.
- **Não são exclusivas:** em produção dá para combinar (ex.: ColBERT + Time-Aware +
  Compressão no mesmo pipeline).
