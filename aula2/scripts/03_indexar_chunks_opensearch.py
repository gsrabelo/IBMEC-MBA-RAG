"""
03_indexar_chunks_opensearch.py - Indexa PDFs juridicos no OpenSearch (busca vetorial).

Pipeline de ingestao da Aula 2:
  1. Le um PDF (ou uma pasta de PDFs).
  2. Extrai o texto em Markdown com o Docling.
  3. Quebra em chunks (Recursive por padrao; Header opcional).
  4. Gera o embedding de cada chunk com o Ollama.
  5. Grava os chunks + vetores em um indice kNN do OpenSearch (com metadados).

Depois disso, o script 04_naive_rag.py consegue responder perguntas sobre os PDFs.

Precisa do OpenSearch e do Ollama rodando (veja 00_check_ambiente.py).

Uso:
    python 03_indexar_chunks_opensearch.py
    python 03_indexar_chunks_opensearch.py --pdf ../datasets/Manual_DPCA_atualizado.pdf --recriar
    python 03_indexar_chunks_opensearch.py --pasta ../datasets --estrategia recursive
"""

import argparse
from pathlib import Path

import requests
from haystack import Document
from haystack.document_stores.types import DuplicatePolicy
from haystack_integrations.components.embedders.ollama import OllamaDocumentEmbedder
from haystack_integrations.document_stores.opensearch import OpenSearchDocumentStore

import _comum

SEPARADORES_JURIDICOS = ["\n\n", "\n", ". ", "; ", ", ", " ", ""]


def pdf_para_markdown(caminho_pdf):
    """Extrai o texto de um PDF em Markdown usando o Docling."""
    from docling.document_converter import DocumentConverter

    return DocumentConverter().convert(str(caminho_pdf)).document.export_to_markdown()


def quebrar_em_chunks(markdown, estrategia, tamanho, overlap):
    """Quebra o Markdown em pedacos. Devolve lista de (texto, metadados)."""
    if estrategia == "header":
        from langchain_text_splitters import MarkdownHeaderTextSplitter

        splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[("#", "Secao"), ("##", "Subsecao"), ("###", "Item")],
            strip_headers=False,
        )
        docs = splitter.split_text(markdown)
        return [(d.page_content, dict(d.metadata)) for d in docs]

    # padrao: recursive
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        separators=SEPARADORES_JURIDICOS, chunk_size=tamanho, chunk_overlap=overlap,
        length_function=len,
    )
    return [(t, {}) for t in splitter.split_text(markdown)]


def coletar_pdfs(args):
    """Decide a lista de PDFs a processar a partir dos argumentos."""
    if args.pasta:
        return sorted(Path(args.pasta).glob("*.pdf"))
    return [Path(args.pdf)]


def apagar_indice(url, indice, auth):
    try:
        r = requests.delete(f"{url}/{indice}", auth=auth, timeout=10)
        if r.status_code in (200, 404):
            print(f"Indice '{indice}' apagado (ou ja nao existia).")
    except Exception as e:
        print(f"Aviso: nao consegui apagar o indice: {e}")


def main():
    parser = argparse.ArgumentParser(description="Indexa PDFs no OpenSearch (chunks + kNN).")
    parser.add_argument("--pdf", default=str(_comum.PDF_DIGITAL), help="PDF a indexar")
    parser.add_argument("--pasta", default=None, help="indexa TODOS os PDFs desta pasta")
    parser.add_argument("--indice", default="aula2_juridico", help="nome do indice")
    parser.add_argument("--estrategia", default="recursive", choices=["recursive", "header"],
                        help="estrategia de chunking (padrao: recursive)")
    parser.add_argument("--chunk-size", type=int, default=800, help="tamanho do chunk (chars)")
    parser.add_argument("--overlap", type=int, default=250, help="sobreposicao entre chunks (chars)")
    parser.add_argument("--modelo", default=None, help="modelo de embedding (padrao: .env)")
    parser.add_argument("--recriar", action="store_true", help="apaga o indice antes de indexar")
    args = parser.parse_args()

    _comum.carregar_env()
    base_url, modelo_padrao = _comum.config_ollama()
    modelo = args.modelo or modelo_padrao
    os_cfg = _comum.config_opensearch()
    auth = (os_cfg["usuario"], os_cfg["senha"]) if os_cfg["usuario"] else None
    dim = _comum.dimensao_do_modelo(modelo)

    print("=" * 60)
    print("  INDEXACAO DE CHUNKS NO OPENSEARCH - Aula 2")
    print("=" * 60)
    print(f"Indice: {args.indice} | Chunking: {args.estrategia} | Modelo: {modelo} (dim {dim})")

    if args.recriar:
        apagar_indice(os_cfg["url"], args.indice, auth)

    store = OpenSearchDocumentStore(
        hosts=os_cfg["url"], index=args.indice, embedding_dim=dim,
        http_auth=auth, use_ssl=False, verify_certs=False,
    )

    # Processa cada PDF: extrai -> chunk -> vira Document do Haystack.
    documentos = []
    for caminho in coletar_pdfs(args):
        if not caminho.exists():
            print(f"  PDF nao encontrado, pulando: {caminho}")
            continue
        print(f"\nProcessando: {caminho.name}")
        markdown = pdf_para_markdown(caminho)
        pedacos = quebrar_em_chunks(markdown, args.estrategia, args.chunk_size, args.overlap)
        print(f"  {len(pedacos)} chunks gerados")
        for i, (texto, meta) in enumerate(pedacos):
            meta = {**meta, "fonte": caminho.name, "chunk_id": i}
            documentos.append(Document(content=texto, meta=meta))

    if not documentos:
        print("\nNenhum chunk para indexar. Verifique os caminhos dos PDFs.")
        return

    print(f"\nGerando embeddings de {len(documentos)} chunks via Ollama...")
    embedder = OllamaDocumentEmbedder(model=modelo, url=base_url)
    docs_com_vetor = embedder.run(documents=documentos)["documents"]

    print("Gravando no OpenSearch...")
    qtd = store.write_documents(docs_com_vetor, policy=DuplicatePolicy.OVERWRITE)
    print(f"\nPronto! {qtd} chunks indexados no indice '{args.indice}'.")
    print("Agora rode: python 04_naive_rag.py --pergunta \"sua pergunta aqui\"")


if __name__ == "__main__":
    main()
