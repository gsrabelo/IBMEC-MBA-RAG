"""
05_multimodal_clip.py - Multimodal RAG sobre PDFs: OCR + CLIP + ColPali + OpenSearch.

Para cada PDF em datasets/pdfs/:
  1) renderiza CADA PAGINA como imagem (PyMuPDF);
  2) extrai o TEXTO de cada pagina via DOCLING com OCR (campo 'conteudo');
  3) indexa no OpenSearch em DOIS indices:
       - visual: embedding da IMAGEM da pagina pelo CLIP (clip-ViT-B-32, 512d)
       - texto : embedding do OCR ('conteudo') pelo Ollama (nomic, 768d)
  4) para ColPali, indexa o PDF com o byaldi (late interaction multimodal).

MODOS de busca (--modo):
  - visual  : CLIP (imagem) - bom para achar FIGURAS por descricao visual.
  - texto   : OCR + embedder de texto - bom para o CONTEUDO textual da pagina.
  - hibrido : funde visual + texto por RRF (padrao).
  - colpali : ColPali/byaldi - estado da arte em recuperar PAGINA por texto.

Por que os 4: o CLIP "puro" e fraco em pagina cheia de texto (ve categorias visuais,
nao le bem o texto); o texto via OCR cobre a semantica textual; o ColPali (late
interaction visual) e o que melhor recupera paginas de documento por consulta textual.

Precisa: pip install sentence-transformers pillow pymupdf docling   (visual/texto/hibrido)
         pip install byaldi                                          (colpali; puxa torch)
Sem PDFs em datasets/pdfs/, gera um exemplo (4 paginas).

Uso:
    python 05_multimodal_clip.py --indexar
    python 05_multimodal_clip.py --consulta "mapa de calor de criminalidade" --modo hibrido
    python 05_multimodal_clip.py --consulta "..." --modo colpali
    python 05_multimodal_clip.py --indexar --recriar
"""

import argparse
from pathlib import Path

import requests

import _comum

_comum.carregar_env()

PASTA_PDFS = _comum.PASTA_DATASETS / "pdfs"
PASTA_IMG = _comum.PASTA_DATASETS / "paginas_png"
INDICE_VISUAL = "aula11_mm_visual"   # CLIP, 512d
INDICE_TEXTO = "aula11_mm_texto"     # OCR + Ollama, 768d
INDICE_COLPALI = "aula11_colpali"    # byaldi
CLIP_DIM = 512

_CLIP = None


def clip_modelo():
    global _CLIP
    if _CLIP is None:
        from sentence_transformers import SentenceTransformer
        _CLIP = SentenceTransformer("clip-ViT-B-32")
    return _CLIP


def _store(indice, dim):
    from haystack_integrations.document_stores.opensearch import OpenSearchDocumentStore

    os_cfg = _comum.config_opensearch()
    auth = (os_cfg["usuario"], os_cfg["senha"]) if os_cfg["usuario"] else None
    return OpenSearchDocumentStore(hosts=os_cfg["url"], index=indice, embedding_dim=dim,
                                   http_auth=auth, use_ssl=False, verify_certs=False)


def store_visual():
    return _store(INDICE_VISUAL, CLIP_DIM)


def store_texto():
    _, modelo = _comum.config_ollama()
    return _store(INDICE_TEXTO, _comum.dimensao_do_modelo(modelo))


# ---------------------------------------------------------------------------
# PDF de exemplo (paginas com imagem + legenda para OCR), montado com PyMuPDF
# ---------------------------------------------------------------------------
def gerar_pdf_exemplo():
    from PIL import Image, ImageDraw
    try:
        import fitz
    except ImportError:
        import pymupdf as fitz

    PASTA_PDFS.mkdir(parents=True, exist_ok=True)
    PASTA_IMG.mkdir(parents=True, exist_ok=True)
    legendas = [
        "Mapa de calor de criminalidade - Sao Paulo 2024",
        "Organograma da organizacao criminosa investigada",
        "Tabela de apreensoes por municipio",
        "Grafico de evolucao de ocorrencias 2020-2024",
    ]
    import random
    random.seed(1)
    pngs = []
    for i, leg in enumerate(legendas):
        img = Image.new("RGB", (700, 500), "white")
        d = ImageDraw.Draw(img)
        d.text((20, 20), leg, fill="black")
        if i == 0:
            for a in range(10):
                for b in range(8):
                    v = random.randint(0, 255)
                    d.rectangle([60+a*55, 80+b*45, 60+a*55+55, 80+b*45+45], fill=(255, 255-v, 255-v))
        elif i == 1:
            for c in [(320, 80, 400, 120), (200, 220, 280, 260), (440, 220, 520, 260)]:
                d.rectangle(c, outline="black", width=3, fill="#cfe8ff")
            d.line([360, 120, 240, 220], fill="black", width=3)
            d.line([360, 120, 480, 220], fill="black", width=3)
        elif i == 2:
            for x in range(60, 641, 96):
                d.line([x, 80, x, 460], fill="black", width=2)
            for y in range(80, 461, 60):
                d.line([60, y, 640, y], fill="black", width=2)
        else:
            d.line([60, 460, 60, 80], fill="black", width=2)
            d.line([60, 460, 640, 460], fill="black", width=2)
            d.line([(60, 440), (200, 360), (340, 380), (480, 220), (640, 120)], fill="blue", width=4)
        png = PASTA_IMG / f"_exemplo_p{i+1}.png"
        img.save(png)
        pngs.append(png)
    saida = PASTA_PDFS / "exemplo_multimodal.pdf"
    doc = fitz.open()
    for png in pngs:
        pix = fitz.Pixmap(str(png))
        page = doc.new_page(width=pix.width, height=pix.height)
        page.insert_image(page.rect, filename=str(png))
    doc.save(str(saida))
    doc.close()
    print(f"PDF de exemplo gerado: {saida} (4 paginas)")
    return [saida]


def listar_pdfs():
    PASTA_PDFS.mkdir(parents=True, exist_ok=True)
    return sorted(PASTA_PDFS.glob("*.pdf")) or gerar_pdf_exemplo()


def renderizar_paginas(pdf_path):
    try:
        import fitz
    except ImportError:
        import pymupdf as fitz

    PASTA_IMG.mkdir(parents=True, exist_ok=True)
    out = []
    doc = fitz.open(pdf_path)
    for i, page in enumerate(doc, 1):
        pix = page.get_pixmap(dpi=150)
        caminho = PASTA_IMG / f"{pdf_path.stem}_p{i:02d}.png"
        pix.save(caminho)
        out.append((i, caminho))
    doc.close()
    return out


def texto_por_pagina(pdf_path):
    """Texto por pagina via Docling (OCR). Fallback: text layer do PyMuPDF."""
    try:
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption

        opts = PdfPipelineOptions()
        opts.do_ocr = True
        opts.do_table_structure = True
        conv = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)})
        doc = conv.convert(str(pdf_path)).document
        paginas = {}
        for item in getattr(doc, "texts", []):
            txt = getattr(item, "text", "") or ""
            for prov in (getattr(item, "prov", None) or []):
                pno = getattr(prov, "page_no", None)
                if pno is not None:
                    paginas.setdefault(pno, []).append(txt)
        if paginas:
            return {p: " ".join(t).strip() for p, t in paginas.items()}
    except Exception as e:
        print(f"  (Docling falhou: {e}; usando o text layer do PDF)")
    try:
        import fitz
    except ImportError:
        import pymupdf as fitz
    doc = fitz.open(pdf_path)
    res = {i: page.get_text().strip() for i, page in enumerate(doc, 1)}
    doc.close()
    return res


# ---------------------------------------------------------------------------
# Indexacao
# ---------------------------------------------------------------------------
def indexar_opensearch(recriar):
    """Indexa cada pagina nos indices VISUAL (CLIP) e TEXTO (OCR+Ollama)."""
    from haystack import Document
    from PIL import Image

    os_cfg = _comum.config_opensearch()
    auth = (os_cfg["usuario"], os_cfg["senha"]) if os_cfg["usuario"] else None
    if recriar:
        for idx in (INDICE_VISUAL, INDICE_TEXTO):
            try:
                requests.delete(f"{os_cfg['url']}/{idx}", auth=auth, timeout=10)
            except Exception:
                pass

    sv, st = store_visual(), store_texto()
    clip = clip_modelo()
    temb = _comum.text_embedder()
    if hasattr(temb, "warm_up"):
        temb.warm_up()

    docs_visual, docs_texto = [], []
    for pdf in listar_pdfs():
        print(f"\nProcessando {pdf.name} ...")
        paginas = renderizar_paginas(pdf)
        textos = texto_por_pagina(pdf)
        for pagina, png in paginas:
            conteudo = textos.get(pagina, "") or "(sem texto)"
            meta = {"id_original": f"{pdf.stem}_p{pagina}", "pdf": pdf.name,
                    "pagina": pagina, "caminho_imagem": str(png), "conteudo": conteudo}
            # visual (CLIP imagem)
            v = clip.encode([Image.open(png)], normalize_embeddings=True)[0]
            docs_visual.append(Document(content=conteudo, embedding=v.tolist(), meta=dict(meta)))
            # texto (OCR -> Ollama)
            t = temb.run(text=conteudo)["embedding"]
            docs_texto.append(Document(content=conteudo, embedding=t, meta=dict(meta)))
            print(f"  pagina {pagina}: OCR {len(conteudo)} chars | CLIP+texto ok")
    sv.write_documents(docs_visual)
    st.write_documents(docs_texto)
    print(f"\nOK - {len(docs_visual)} paginas em '{INDICE_VISUAL}' (CLIP) e '{INDICE_TEXTO}' (texto).")


def indexar_colpali(recriar):
    """Indexa os PDFs com o ColPali (byaldi, late interaction multimodal)."""
    try:
        from byaldi import RAGMultiModalModel
    except ImportError:
        print("[ERRO] instale: pip install byaldi")
        return
    listar_pdfs()  # garante pelo menos o PDF de exemplo
    print("Carregando ColPali (vidore/colpali-v1.2) e indexando... (baixa o modelo na 1a vez)")
    model = RAGMultiModalModel.from_pretrained("vidore/colpali-v1.2")
    model.index(input_path=str(PASTA_PDFS), index_name=INDICE_COLPALI,
                store_collection_with_index=False, overwrite=bool(recriar))
    print(f"OK - PDFs indexados no ColPali (indice '{INDICE_COLPALI}').")


# ---------------------------------------------------------------------------
# Busca por modo
# ---------------------------------------------------------------------------
def _knn(store, query_embedding, top_k):
    from haystack_integrations.components.retrievers.opensearch import (
        OpenSearchEmbeddingRetriever,
    )
    return OpenSearchEmbeddingRetriever(document_store=store, top_k=top_k).run(
        query_embedding=query_embedding)["documents"]


def buscar_visual(consulta, top_k):
    emb = clip_modelo().encode([consulta], normalize_embeddings=True)[0].tolist()
    return _knn(store_visual(), emb, top_k)


def buscar_texto(consulta, top_k):
    temb = _comum.text_embedder()
    if hasattr(temb, "warm_up"):
        temb.warm_up()
    emb = temb.run(text=consulta)["embedding"]
    return _knn(store_texto(), emb, top_k)


def buscar_hibrido(consulta, top_k):
    """Funde visual + texto por Reciprocal Rank Fusion (RRF)."""
    vis = buscar_visual(consulta, top_k)
    txt = buscar_texto(consulta, top_k)
    scores, por_id = {}, {}
    for lista in (vis, txt):
        for rank, d in enumerate(lista):
            i = d.meta.get("id_original")
            por_id[i] = d
            scores[i] = scores.get(i, 0.0) + 1.0 / (60 + rank + 1)
    ordenados = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [por_id[i] for i, _ in ordenados]


def imprimir(docs, consulta):
    print(f"\nConsulta: {consulta!r}")
    for i, d in enumerate(docs, 1):
        sc = f" (score={d.score:.3f})" if d.score is not None else ""
        print(f"  {i}. {d.meta.get('pdf')} p.{d.meta.get('pagina')}{sc}")
        print(f"      imagem: {d.meta.get('caminho_imagem')}")
        print(f"      OCR: {(d.meta.get('conteudo') or '')[:90]}")


def buscar_colpali(consulta, top_k):
    try:
        from byaldi import RAGMultiModalModel
    except ImportError:
        print("[ERRO] instale: pip install byaldi")
        return
    model = RAGMultiModalModel.from_index(INDICE_COLPALI)
    res = model.search(consulta, k=top_k)
    print(f"\nConsulta (ColPali): {consulta!r}")
    for i, r in enumerate(res, 1):
        print(f"  {i}. doc={getattr(r, 'doc_id', '?')} pagina={getattr(r, 'page_num', '?')} "
              f"score={getattr(r, 'score', 0):.3f}")


def buscar(consulta, modo, top_k):
    if modo == "visual":
        imprimir(buscar_visual(consulta, top_k), consulta)
    elif modo == "texto":
        imprimir(buscar_texto(consulta, top_k), consulta)
    elif modo == "colpali":
        buscar_colpali(consulta, top_k)
    else:  # hibrido
        imprimir(buscar_hibrido(consulta, top_k), consulta)


def main():
    parser = argparse.ArgumentParser(description="Multimodal RAG: OCR+CLIP+ColPali (Aula 11).")
    parser.add_argument("--indexar", action="store_true")
    parser.add_argument("--recriar", action="store_true")
    parser.add_argument("--consulta", default=None)
    parser.add_argument("--modo", default="hibrido", choices=["visual", "texto", "hibrido", "colpali"])
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    print("=" * 60)
    print(f"  MULTIMODAL RAG (modo: {args.modo}) - Aula 11")
    print("=" * 60)
    try:
        if args.indexar or args.recriar:
            if args.modo == "colpali":
                indexar_colpali(args.recriar)
            else:
                indexar_opensearch(args.recriar)
        if args.consulta:
            buscar(args.consulta, args.modo, args.top_k)
        if not args.indexar and not args.recriar and not args.consulta:
            for q in ["mapa de calor de criminalidade", "organograma da faccao",
                      "tabela de apreensoes", "grafico de evolucao de ocorrencias"]:
                buscar(q, args.modo, args.top_k)
    except ImportError as e:
        print(f"\n[ERRO] dependencia faltando: {e}")


if __name__ == "__main__":
    main()
