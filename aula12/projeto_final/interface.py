"""
interface.py - Interface Gradio para a API do Projeto Final (Aula 12).

Facilita o uso da API sem curl/Postman. Tem duas abas:
  1) Ingestao  - envia um documento e mostra o RELATORIO da decisao
                 (tecnica de extracao do agente + destino + tecnica de chunking + motivos).
  2) Consulta  - faz uma pergunta e mostra a resposta RAG + fontes.

A interface NAO faz logica de RAG: ela so chama a API (separacao clara
"motor" x "interface"). Configure a URL da API em API_URL (.env), padrao http://localhost:8000.

Rodar (com a API ja no ar):
    python interface.py        # abre em http://localhost:7860
"""

import os

import gradio as gr
import requests
from dotenv import load_dotenv

load_dotenv()
API_URL = os.getenv("API_URL", "http://localhost:8001")
GRADIO_PORT = int(os.getenv("GRADIO_PORT", "7860"))
API_KEY = os.getenv("API_KEY", "")  # so se a API exigir X-API-Key

CHUNKING_OPCOES = ["auto", "fixo", "recursivo", "sentenca_janela", "semantico", "hierarquico"]
DESTINO_OPCOES = ["auto", "opensearch", "grafo"]


def _headers():
    return {"X-API-Key": API_KEY} if API_KEY else {}


# ---------------------------------------------------------------------------
# Aba 1 - Ingestao
# ---------------------------------------------------------------------------
def ingerir(arquivo, estrategia, chunking):
    if not arquivo:
        return "Selecione um arquivo.", {}
    try:
        with open(arquivo, "rb") as f:
            resp = requests.post(
                f"{API_URL}/ingestao",
                params={"estrategia": estrategia, "chunking": chunking},
                files={"arquivo": (os.path.basename(arquivo), f)},
                headers=_headers(), timeout=600)
        resp.raise_for_status()
        dados = resp.json()
    except Exception as e:
        return f"Erro ao chamar a API: {e}", {}

    if not dados.get("ok"):
        return f"Falha na ingestao: {dados.get('erro')}", dados

    r = dados["relatorio"]
    resumo = (
        f"### Decisao da ingestao\n"
        f"- **Arquivo:** {r['arquivo']} ({r['n_caracteres']} caracteres)\n"
        f"- **Complexidade:** {r['complexidade']}\n"
        f"- **Extracao (agente):** `{r['tecnica_extracao']}` — {r.get('motivo_extracao','')}\n"
        f"- **Destino:** `{r['destino']}` — {r.get('motivo_destino','')}\n"
        f"- **Chunking:** `{r.get('chunking','')}` — {r.get('motivo_chunking','')}\n"
        f"- **Chunks indexados:** {r['n_chunks']}\n"
    )
    return resumo, dados


# ---------------------------------------------------------------------------
# Aba 2 - Consulta
# ---------------------------------------------------------------------------
def consultar(pergunta, destino, top_k):
    if not pergunta.strip():
        return "Digite uma pergunta.", {}
    try:
        resp = requests.post(
            f"{API_URL}/consulta",
            json={"pergunta": pergunta, "destino": destino, "top_k": int(top_k)},
            headers=_headers(), timeout=600)
        resp.raise_for_status()
        dados = resp.json()
    except Exception as e:
        return f"Erro ao chamar a API: {e}", {}

    fontes = "\n".join(f"- {f}" for f in dados.get("fontes", [])) or "(sem fontes)"
    texto = (f"### Resposta (destino: {dados.get('destino_usado','?')})\n\n"
             f"{dados.get('resposta','')}\n\n**Fontes:**\n{fontes}")
    return texto, dados


def status():
    try:
        return requests.get(f"{API_URL}/health", headers=_headers(), timeout=30).json()
    except Exception as e:
        return {"erro": str(e), "dica": f"A API esta no ar em {API_URL}?"}


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
with gr.Blocks(title="Projeto Final - RAG Juridico (Aula 12)") as demo:
    gr.Markdown(
        "# Projeto Final — Ingestao Inteligente + RAG\n"
        "Interface da API: o **agente** decide a extracao e a **heuristica** decide o "
        "**destino** (OpenSearch/grafo) e a **tecnica de chunking**. "
        f"API: `{API_URL}`")

    with gr.Tab("1) Ingestao"):
        arquivo = gr.File(label="Documento (PDF, DOCX, XLSX, imagem, TXT)", type="filepath")
        with gr.Row():
            estrategia = gr.Dropdown(DESTINO_OPCOES, value="auto", label="Destino (override)")
            chunking = gr.Dropdown(CHUNKING_OPCOES, value="auto", label="Chunking (override)")
        btn_ing = gr.Button("Ingerir", variant="primary")
        out_ing_md = gr.Markdown()
        out_ing_json = gr.JSON(label="Relatorio completo (JSON)")
        btn_ing.click(ingerir, [arquivo, estrategia, chunking], [out_ing_md, out_ing_json])

    with gr.Tab("2) Consulta"):
        pergunta = gr.Textbox(label="Pergunta", lines=2, placeholder="Ex.: Qual o prazo do recurso?")
        with gr.Row():
            destino = gr.Dropdown(DESTINO_OPCOES, value="auto", label="Buscar em")
            top_k = gr.Slider(1, 15, value=5, step=1, label="top_k")
        btn_q = gr.Button("Perguntar", variant="primary")
        out_q_md = gr.Markdown()
        out_q_json = gr.JSON(label="Resposta completa (JSON)")
        btn_q.click(consultar, [pergunta, destino, top_k], [out_q_md, out_q_json])

    with gr.Tab("Status"):
        btn_s = gr.Button("Checar /health")
        out_s = gr.JSON()
        btn_s.click(status, None, out_s)

if __name__ == "__main__":
    demo.launch(server_port=GRADIO_PORT)
