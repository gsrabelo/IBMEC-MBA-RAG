"""
01_ingestao_docling.py - Ingestao de PDF juridico com Docling.

Extrai o texto de um PDF de duas formas e compara:
  1. PyPDF2  -> extracao simples (baseline; perde tabelas e estrutura)
  2. Docling -> converte para Markdown estruturado (mantem titulos e tabelas)

Para PDFs escaneados (imagem de texto), use --ocr para ligar o OCR do Docling.

Uso:
    python 01_ingestao_docling.py
    python 01_ingestao_docling.py --pdf ../datasets/Laudo-Minimal.pdf --ocr
    python 01_ingestao_docling.py --salvar
"""

import argparse
import time
from pathlib import Path

import _comum


def extrair_pypdf2(caminho_pdf):
    """Extrai o texto de um PDF com PyPDF2 (juntando todas as paginas)."""
    import PyPDF2

    with open(caminho_pdf, "rb") as arquivo:
        leitor = PyPDF2.PdfReader(arquivo)
        partes = [(pag.extract_text() or "") for pag in leitor.pages]
    return "\n".join(partes)


def extrair_docling(caminho_pdf, usar_ocr):
    """Converte o PDF para Markdown estruturado usando o Docling."""
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    opcoes = PdfPipelineOptions()
    opcoes.do_ocr = usar_ocr            # liga o OCR (PDFs escaneados)
    opcoes.do_table_structure = True    # tenta reconstruir tabelas

    print("Inicializando Docling (pode baixar modelos na 1a vez)...")
    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opcoes)}
    )
    resultado = converter.convert(caminho_pdf)
    return resultado.document.export_to_markdown()


def comparar(texto_pypdf, markdown_docling):
    print("\nCOMPARACAO: PyPDF2 vs Docling")
    print("-" * 55)
    print(f"{'Metrica':<28}{'PyPDF2':>12}{'Docling':>12}")
    print(f"{'Caracteres':<28}{len(texto_pypdf):>12}{len(markdown_docling):>12}")
    print(f"{'Palavras':<28}{len(texto_pypdf.split()):>12}{len(markdown_docling.split()):>12}")
    tem = lambda t, s: "sim" if s in t else "nao"
    print(f"{'Tem tabela (|)':<28}{tem(texto_pypdf,'|'):>12}{tem(markdown_docling,'|'):>12}")
    print(f"{'Tem titulos (#)':<28}{tem(texto_pypdf,'#'):>12}{tem(markdown_docling,'#'):>12}")
    print("\nDocling preserva estrutura (titulos e tabelas); PyPDF2 so o texto cru.")


def main():
    parser = argparse.ArgumentParser(description="Ingestao de PDF com Docling.")
    parser.add_argument("--pdf", default=str(_comum.PDF_DIGITAL),
                        help="caminho do PDF (padrao: Manual_DPCA_atualizado.pdf)")
    parser.add_argument("--ocr", action="store_true",
                        help="liga o OCR do Docling (para PDFs escaneados)")
    parser.add_argument("--salvar", action="store_true",
                        help="salva o Markdown do Docling em um arquivo .md")
    args = parser.parse_args()

    caminho = Path(args.pdf)
    print("=" * 60)
    print("  INGESTAO COM DOCLING - Aula 2")
    print("=" * 60)
    if not caminho.exists():
        print(f"PDF nao encontrado: {caminho}")
        return
    print(f"PDF: {caminho.name} | OCR: {'ligado' if args.ocr else 'desligado'}")

    print("\nExtraindo com PyPDF2 (baseline)...")
    texto_pypdf = extrair_pypdf2(caminho)

    inicio = time.time()
    markdown = extrair_docling(caminho, args.ocr)
    print(f"Docling concluido em {time.time() - inicio:.1f}s")

    print("\nPREVIEW DO MARKDOWN (Docling, primeiros 800 chars):")
    print("-" * 55)
    print(markdown[:800])

    comparar(texto_pypdf, markdown)

    if args.salvar:
        saida = _comum.PASTA_SCRIPTS / f"{caminho.stem}.md"
        saida.write_text(markdown, encoding="utf-8")
        print(f"\nMarkdown salvo em: {saida}")


if __name__ == "__main__":
    main()
