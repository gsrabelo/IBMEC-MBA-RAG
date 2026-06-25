"""
extracao.py - Extracao INTELIGENTE de documentos (Agente Haystack decide a tecnica).

Fluxo:
  1) probe(): le 'sinais' baratos do documento (extensao, paginas, texto, imagens).
  2) Um AGENTE Haystack (LLM via Groq, tool-calling) recebe esses sinais e CHAMA a
     ferramenta de extracao mais adequada:
       - extrair_planilha : XLSX/CSV (pandas)
       - extrair_texto    : PDF nativo/DOCX/TXT com camada de texto (Docling, sem OCR)
       - extrair_com_ocr  : PDF escaneado / imagens / paginas com figuras (Docling + OCR)
  3) As ferramentas guardam o conteudo extraido num cache e devolvem ao agente apenas um
     RESUMO curto (assim o conteudo grande nao infla o contexto do LLM).

Lazy imports: docling/pandas/fitz so sao carregados quando a ferramenta roda.
"""

import json
from pathlib import Path

from haystack.dataclasses import ChatMessage

from . import config
from .log import obter_logger

log = obter_logger(__name__)

# cache: caminho -> {"conteudo": str, "tabelas": list, "tecnica": str}
_CACHE = {}

EXT_PLANILHA = {".xlsx", ".xls", ".csv", ".tsv"}
EXT_IMAGEM = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp"}
EXT_TEXTO = {".txt", ".md", ".html", ".htm"}


# ---------------------------------------------------------------------------
# 1) Sinais do documento (probe barato, sem LLM)
# ---------------------------------------------------------------------------
def probe(caminho):
    p = Path(caminho)
    ext = p.suffix.lower()
    sinais = {"extensao": ext, "n_paginas": 0, "n_chars_texto": 0,
              "tem_imagens": False, "eh_planilha": ext in EXT_PLANILHA,
              "eh_imagem": ext in EXT_IMAGEM}
    if ext == ".pdf":
        try:
            try:
                import fitz
            except ImportError:
                import pymupdf as fitz
            doc = fitz.open(caminho)
            sinais["n_paginas"] = doc.page_count
            chars, imgs = 0, 0
            for page in doc:
                chars += len(page.get_text())
                imgs += len(page.get_images())
            doc.close()
            sinais["n_chars_texto"] = chars
            sinais["tem_imagens"] = imgs > 0
            # heuristica: pouco texto + imagens => provavelmente escaneado
            sinais["provavel_escaneado"] = chars < 100 and imgs > 0
        except Exception as e:
            sinais["erro_probe"] = str(e)
    return sinais


# ---------------------------------------------------------------------------
# Implementacoes de extracao (chamadas pelas ferramentas do agente)
# ---------------------------------------------------------------------------
def _docling_markdown(caminho, com_ocr):
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    opts = PdfPipelineOptions()
    opts.do_ocr = com_ocr
    opts.do_table_structure = True
    conv = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)})
    return conv.convert(str(caminho)).document.export_to_markdown()


def _impl_planilha(caminho):
    import pandas as pd

    p = Path(caminho)
    partes, tabelas = [], []
    if p.suffix.lower() in {".csv", ".tsv"}:
        sep = "\t" if p.suffix.lower() == ".tsv" else ","
        planilhas = {"dados": pd.read_csv(caminho, sep=sep)}
    else:
        planilhas = pd.read_excel(caminho, sheet_name=None)  # todas as abas
    for nome, df in planilhas.items():
        md = df.to_markdown(index=False)
        partes.append(f"## Aba: {nome}\n{md}")
        tabelas.append({"aba": nome, "linhas": len(df), "colunas": list(df.columns)})
    return "\n\n".join(partes), tabelas


def _impl_texto(caminho):
    ext = Path(caminho).suffix.lower()
    if ext in EXT_TEXTO:
        return Path(caminho).read_text(encoding="utf-8", errors="ignore"), []
    # PDF/DOCX via Docling sem OCR
    return _docling_markdown(caminho, com_ocr=False), []


def _impl_ocr(caminho):
    # PDF escaneado / imagem / figuras -> Docling com OCR
    return _docling_markdown(caminho, com_ocr=True), []


def _guardar(caminho, conteudo, tabelas, tecnica):
    _CACHE[caminho] = {"conteudo": conteudo or "", "tabelas": tabelas or [], "tecnica": tecnica}
    log.info("Extracao concluida: tecnica=%s, %d caracteres, %d tabela(s)",
             tecnica, len(conteudo or ""), len(tabelas or []))
    return json.dumps({"tecnica": tecnica, "n_caracteres": len(conteudo or ""),
                       "n_tabelas": len(tabelas or [])}, ensure_ascii=False)


# ---------------------------------------------------------------------------
# 2) Ferramentas do agente (recebem o caminho, extraem, devolvem RESUMO)
# ---------------------------------------------------------------------------
def extrair_planilha(caminho: str) -> str:
    """Extrai dados de PLANILHAS (XLSX/CSV) como tabelas em markdown."""
    try:
        conteudo, tabelas = _impl_planilha(caminho)
        return _guardar(caminho, conteudo, tabelas, "planilha")
    except Exception as e:
        return f"[erro planilha: {e}]"


def extrair_texto(caminho: str) -> str:
    """Extrai TEXTO de documentos com camada de texto (PDF nativo, DOCX, TXT) - sem OCR."""
    try:
        conteudo, tabelas = _impl_texto(caminho)
        return _guardar(caminho, conteudo, tabelas, "texto")
    except Exception as e:
        return f"[erro texto: {e}]"


def extrair_com_ocr(caminho: str) -> str:
    """Extrai texto de PDFs ESCANEADOS, IMAGENS ou paginas com FIGURAS, usando OCR."""
    try:
        conteudo, tabelas = _impl_ocr(caminho)
        return _guardar(caminho, conteudo, tabelas, "ocr")
    except Exception as e:
        return f"[erro ocr: {e}]"


# ---------------------------------------------------------------------------
# 3) Agente Haystack que decide qual ferramenta usar
# ---------------------------------------------------------------------------
SYSTEM_EXTRACAO = (
    "Voce decide a MELHOR tecnica para extrair o conteudo de um documento, a partir dos "
    "sinais fornecidos. Chame UMA ferramenta:\n"
    "- extrair_planilha: se for planilha (.xlsx/.csv/.tsv).\n"
    "- extrair_com_ocr: se for imagem, PDF escaneado (pouco texto + imagens) ou com figuras.\n"
    "- extrair_texto: nos demais casos (PDF nativo, DOCX, TXT com texto).\n"
    "Depois de chamar a ferramenta, responda em uma frase o motivo da escolha."
)


def _param_caminho(desc):
    return {"type": "object",
            "properties": {"caminho": {"type": "string", "description": desc}},
            "required": ["caminho"]}


def criar_agente_extracao(max_passos=3):
    from haystack.components.agents import Agent
    from haystack.components.generators.chat import OpenAIChatGenerator
    from haystack.tools import Tool
    from haystack.utils import Secret

    api_key, modelo, base_url = config.config_groq()
    gerador = OpenAIChatGenerator(api_key=Secret.from_token(api_key), model=modelo,
                                  api_base_url=base_url,
                                  generation_kwargs={"temperature": 0.0, "max_tokens": 300})
    tools = [
        Tool(name="extrair_planilha", description="Extrai planilhas XLSX/CSV/TSV.",
             parameters=_param_caminho("caminho do arquivo"), function=extrair_planilha),
        Tool(name="extrair_texto", description="Extrai texto de PDF nativo/DOCX/TXT (sem OCR).",
             parameters=_param_caminho("caminho do arquivo"), function=extrair_texto),
        Tool(name="extrair_com_ocr", description="Extrai via OCR (PDF escaneado/imagem/figuras).",
             parameters=_param_caminho("caminho do arquivo"), function=extrair_com_ocr),
    ]
    return Agent(chat_generator=gerador, tools=tools, system_prompt=SYSTEM_EXTRACAO,
                 exit_conditions=["text"], max_agent_steps=max_passos)


def _tecnica_chamada(mensagens):
    for m in mensagens:
        for tc in (getattr(m, "tool_calls", None) or []):
            return getattr(tc, "tool_name", None) or getattr(tc, "name", None)
    return None


MAPA_COMPLEXIDADE = {"planilha": "planilha", "ocr": "complexo", "texto": "texto_simples"}

# tecnica -> funcao de extracao (usada pelo fallback heuristico)
_IMPL = {"planilha": _impl_planilha, "ocr": _impl_ocr, "texto": _impl_texto}


def escolher_por_sinais(sinais):
    """Escolha DETERMINISTICA da tecnica a partir do probe (fallback, sem LLM)."""
    if sinais.get("eh_planilha"):
        return "planilha"
    if sinais.get("eh_imagem") or sinais.get("provavel_escaneado"):
        return "ocr"
    return "texto"


def _extrair_direto(caminho, tecnica):
    """Roda a extracao da tecnica escolhida e popula o cache (sem agente)."""
    conteudo, tabelas = _IMPL.get(tecnica, _impl_texto)(caminho)
    _guardar(caminho, conteudo, tabelas, tecnica)
    return _CACHE[caminho]


def decidir_e_extrair(caminho):
    """Roda o agente, descobre a tecnica usada e devolve (sinais, tecnica, complexidade, motivo, dados).

    O AGENTE (LLM) e o decisor primario. Se ele falhar (ex.: o Groq devolve
    'tool_use_failed' ao parsear o tool-call do llama-3.3) ou nao chamar nenhuma
    ferramenta, caimos num FALLBACK HEURISTICO deterministico baseado nos sinais do
    probe - assim a ingestao nunca quebra por instabilidade do tool-calling.
    """
    log.info("Iniciando extracao: %s", caminho)
    sinais = probe(caminho)
    log.debug("Sinais do probe: %s", json.dumps(sinais, ensure_ascii=False))
    tecnica, motivo = None, ""
    try:
        log.info("Consultando o AGENTE (Groq) para escolher a tecnica de extracao...")
        agente = criar_agente_extracao()
        prompt = (f"Arquivo: {caminho}\nSinais do documento: {json.dumps(sinais, ensure_ascii=False)}\n"
                  "Escolha e chame a ferramenta de extracao adequada.")
        resultado = agente.run(messages=[ChatMessage.from_user(prompt)])
        tecnica = _tecnica_chamada(resultado.get("messages", []))
        motivo = resultado["last_message"].text if resultado.get("last_message") else ""
        log.info("Agente escolheu a tecnica: %s", tecnica or "(nenhuma ferramenta chamada)")
        log.debug("Mensagem final do agente: %s", motivo)
    except Exception as e:
        log.warning("Agente falhou (%s) -> usando fallback heuristico", e)
        motivo = f"(fallback heuristico: agente falhou - {e})"

    dados = _CACHE.get(caminho)
    # fallback: agente nao escolheu/extraiu -> decide por sinais e extrai direto
    if not tecnica or not dados or not dados.get("conteudo"):
        tecnica = escolher_por_sinais(sinais)
        log.info("FALLBACK heuristico: tecnica '%s' escolhida pelos sinais do documento", tecnica)
        if not motivo:
            motivo = f"(fallback heuristico: tecnica '{tecnica}' escolhida pelos sinais)"
        dados = _extrair_direto(caminho, tecnica)

    tecnica_final = dados.get("tecnica", tecnica)
    complexidade = MAPA_COMPLEXIDADE.get(tecnica_final, "texto_simples")
    log.info("Extracao finalizada: tecnica=%s, complexidade=%s", tecnica_final, complexidade)
    return sinais, tecnica_final, complexidade, motivo, dados


def limpar_cache(caminho=None):
    if caminho:
        _CACHE.pop(caminho, None)
    else:
        _CACHE.clear()
