"""
02_chunking_comparar.py - Compara estrategias de chunking (divisao de texto).

Chunking e o passo de quebrar um documento em pedacos (chunks) antes de gerar
embeddings. A forma de quebrar muda muito a qualidade do RAG. Aqui comparamos:

  1. Fixed-Size  - corta a cada N caracteres (rapido, mas corta no meio da frase)
  2. Recursive   - tenta cortar em paragrafo > linha > frase (melhor equilibrio)
  3. Semantic    - corta onde o assunto muda (usa embeddings; precisa do Ollama)
  4. Header      - corta por titulos/secoes do documento (precisa de Markdown)

Por padrao usa um acordao de exemplo. Com --pdf, extrai o texto de um PDF (Docling).

Uso:
    python 02_chunking_comparar.py
    python 02_chunking_comparar.py --estrategia recursive --chunk-size 800 --overlap 250
    python 02_chunking_comparar.py --estrategia todas
    python 02_chunking_comparar.py --pdf ../datasets/Manual_DPCA_atualizado.pdf
"""

import argparse

import _comum

# Separadores hierarquicos bons para texto juridico brasileiro (Recursive).
SEPARADORES_JURIDICOS = ["\n\n", "\n", ". ", "; ", ", ", " ", ""]

# Acordao de exemplo (texto puro) - usado quando nao se passa --pdf.
TEXTO_ACORDAO = """EMENTA
HABEAS CORPUS. TRAFICO DE DROGAS. PRISAO PREVENTIVA. REQUISITOS LEGAIS. FUNDAMENTACAO INIDONEA. ORDEM CONCEDIDA.
1. A prisao preventiva constitui medida cautelar de natureza excepcional, somente admissivel quando presentes os requisitos previstos nos artigos 312 e 313 do Codigo de Processo Penal.
2. A fundamentacao da prisao preventiva nao pode se basear exclusivamente na gravidade abstrata do delito ou na mera reproducao dos termos legais.
3. O excesso de prazo na formacao da culpa configura constrangimento ilegal sanavel pela via do habeas corpus.

RELATORIO
Trata-se de habeas corpus impetrado em favor de SICRANO DE TAL contra decisao que decretou a prisao preventiva no Processo no 0001234-56.2024.8.26.0001, por suposta pratica do artigo 33 da Lei no 11.343/2006.
O impetrante alega: (a) decisao sem fundamentacao idonea; (b) excesso de prazo de 120 dias; (c) paciente primario, com residencia fixa e trabalho licito.
O Ministerio Publico opinou pela denegacao da ordem.

FUNDAMENTACAO
A prisao preventiva e medida de ultima ratio, cabivel apenas quando as demais cautelares do artigo 319 do CPP forem inadequadas. No caso, a decisao limitou-se a citar a garantia da ordem publica sem indicar fatos concretos. A jurisprudencia do STJ e pacifica: a gravidade abstrata nao fundamenta a custodia. Ademais, o paciente esta preso ha mais de 120 dias sem justificativa para a demora.

DISPOSITIVO
CONCEDO a ordem para determinar a imediata colocacao do paciente em liberdade, revogando-se a prisao preventiva.
"""

# Mesmo acordao em Markdown (com titulos) - usado pela estrategia Header.
TEXTO_MARKDOWN = """# EMENTA

HABEAS CORPUS. TRAFICO DE DROGAS. PRISAO PREVENTIVA. FUNDAMENTACAO INIDONEA. ORDEM CONCEDIDA.

# RELATORIO

## Das Alegacoes do Impetrante

Decisao sem fundamentacao idonea; excesso de prazo de 120 dias; paciente primario.

## Da Manifestacao do Ministerio Publico

O Ministerio Publico opinou pela denegacao da ordem.

# FUNDAMENTACAO

## Da Prisao Preventiva como Medida Excepcional

Medida de ultima ratio, cabivel apenas quando as cautelares do art. 319 do CPP forem inadequadas.

## Do Excesso de Prazo

Paciente preso ha mais de 120 dias sem justificativa.

# DISPOSITIVO

CONCEDO a ordem, revogando-se a prisao preventiva.
"""


def chunk_fixed(texto, tamanho, overlap):
    from langchain_text_splitters import CharacterTextSplitter

    splitter = CharacterTextSplitter(separator="\n", chunk_size=tamanho,
                                     chunk_overlap=overlap, length_function=len)
    return splitter.split_text(texto)


def chunk_recursive(texto, tamanho, overlap):
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(separators=SEPARADORES_JURIDICOS,
                                              chunk_size=tamanho, chunk_overlap=overlap,
                                              length_function=len)
    return splitter.split_text(texto)


def chunk_semantic(texto):
    from langchain_experimental.text_splitter import SemanticChunker
    from langchain_ollama import OllamaEmbeddings

    base_url, modelo = _comum.config_ollama()
    embeddings = OllamaEmbeddings(model=modelo, base_url=base_url)
    chunker = SemanticChunker(embeddings=embeddings,
                              breakpoint_threshold_type="percentile",
                              breakpoint_threshold_amount=85)
    return chunker.split_text(texto)


def chunk_header(markdown):
    from langchain_text_splitters import MarkdownHeaderTextSplitter

    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "Secao"), ("##", "Subsecao"), ("###", "Item")],
        strip_headers=False,
    )
    docs = splitter.split_text(markdown)
    return [d.page_content for d in docs]


def estatisticas(chunks):
    tamanhos = [len(c) for c in chunks]
    cortes_ruins = sum(1 for c in chunks if c.strip() and not c.strip().endswith((".", "!", "?", ":")))
    return {
        "n": len(chunks),
        "media": int(sum(tamanhos) / len(tamanhos)) if chunks else 0,
        "min": min(tamanhos) if chunks else 0,
        "max": max(tamanhos) if chunks else 0,
        "cortes_ruins": cortes_ruins,
    }


def mostrar(nome, chunks, detalhar):
    st = estatisticas(chunks)
    print(f"\n[{nome}] {st['n']} chunks | media {st['media']} | min {st['min']} | "
          f"max {st['max']} | cortes ruins {st['cortes_ruins']}")
    if detalhar:
        for i, c in enumerate(chunks, 1):
            print(f"  --- chunk {i} ({len(c)} chars) ---")
            print(f"  {c[:160].strip()}...")


def main():
    parser = argparse.ArgumentParser(description="Compara estrategias de chunking.")
    parser.add_argument("--estrategia", default="todas",
                        choices=["fixed", "recursive", "semantic", "header", "todas"],
                        help="qual estrategia usar (padrao: todas)")
    parser.add_argument("--chunk-size", type=int, default=800, help="tamanho do chunk (chars)")
    parser.add_argument("--overlap", type=int, default=250, help="sobreposicao entre chunks (chars)")
    parser.add_argument("--pdf", default=None, help="extrai o texto deste PDF (via Docling) em vez do exemplo")
    args = parser.parse_args()

    print("=" * 60)
    print("  COMPARACAO DE CHUNKING - Aula 2")
    print("=" * 60)

    # Decide a fonte do texto.
    if args.pdf:
        from docling.document_converter import DocumentConverter

        print(f"Extraindo texto de {args.pdf} com Docling...")
        markdown = DocumentConverter().convert(args.pdf).document.export_to_markdown()
        texto, texto_md = markdown, markdown
    else:
        print("Usando o acordao de exemplo (use --pdf para um PDF real).")
        texto, texto_md = TEXTO_ACORDAO, TEXTO_MARKDOWN

    detalhar = args.estrategia != "todas"
    e = args.estrategia
    ts, ov = args.chunk_size, args.overlap

    if e in ("fixed", "todas"):
        mostrar("Fixed-Size", chunk_fixed(texto, ts, ov), detalhar)
    if e in ("recursive", "todas"):
        mostrar("Recursive", chunk_recursive(texto, ts, ov), detalhar)
    if e in ("semantic", "todas"):
        try:
            mostrar("Semantic", chunk_semantic(texto), detalhar)
        except Exception as ex:
            print(f"\n[Semantic] pulado (precisa do Ollama): {ex}")
    if e in ("header", "todas"):
        mostrar("Header-Based", chunk_header(texto_md), detalhar)

    if e == "todas":
        print("\nDica: Recursive costuma ser o melhor equilibrio; Header e otimo apos o Docling.")


if __name__ == "__main__":
    main()
