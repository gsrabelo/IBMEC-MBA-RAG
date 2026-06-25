"""
03_contextual_retrieval.py - Contextual Retrieval (#T09).

Ideia (Anthropic, 2024): antes de gerar o embedding, a gente PREPENDE a cada
documento/chunk um pequeno "contexto" escrito por um LLM, situando o trecho
(qual orgao, qual tema, o que decide). Assim o embedding fica mais informativo
e a busca encontra o documento certo com mais frequencia.

Este script:
  1. Le N documentos do corpus.
  2. Para cada um, pede a Groq um contexto curto (1-2 frases).
  3. Prepende o contexto ao texto e indexa no OpenSearch (indice contextual).
  4. Mostra alguns exemplos de "antes" e "depois".

Depois, voce pode comparar a busca usando:
    python 02_busca_hibrida_rrf.py --indice aula4_contextual --query "..."

Precisa de OpenSearch, Ollama e Groq.

Uso:
    python 03_contextual_retrieval.py
    python 03_contextual_retrieval.py --limite 50 --recriar
"""

import argparse

import requests
from haystack import Document
from haystack.document_stores.types import DuplicatePolicy
from haystack_integrations.components.embedders.ollama import OllamaDocumentEmbedder
from haystack_integrations.document_stores.opensearch import OpenSearchDocumentStore

import _comum

PROMPT_CONTEXTO = (
    "Escreva em 1-2 frases curtas o contexto deste documento juridico para ajudar "
    "uma busca a encontra-lo: diga o orgao/tribunal, o tema e o que se decide. "
    "Responda APENAS com as frases de contexto, sem rotulos.\n\n"
    "Titulo: {titulo}\nTexto: {texto}"
)


def gerar_contexto(cliente, modelo, titulo, texto):
    resp = cliente.chat.completions.create(
        model=modelo,
        messages=[{"role": "user", "content": PROMPT_CONTEXTO.format(titulo=titulo, texto=texto[:1500])}],
        temperature=0.2, max_tokens=120,
    )
    return resp.choices[0].message.content.strip()


def apagar_indice(url, indice, auth):
    try:
        r = requests.delete(f"{url}/{indice}", auth=auth, timeout=10)
        if r.status_code in (200, 404):
            print(f"Indice '{indice}' apagado (ou ja nao existia).")
    except Exception as e:
        print(f"Aviso: nao consegui apagar o indice: {e}")


def main():
    parser = argparse.ArgumentParser(description="Contextual Retrieval (#T09).")
    parser.add_argument("--indice", default="aula4_contextual", help="indice de saida")
    parser.add_argument("--limite", type=int, default=30, help="quantos documentos enriquecer (chamadas LLM)")
    parser.add_argument("--modelo", default=None, help="modelo de embedding (padrao: .env)")
    parser.add_argument("--recriar", action="store_true", help="apaga o indice antes de indexar")
    args = parser.parse_args()

    _comum.carregar_env()
    base_url, modelo_padrao = _comum.config_ollama()
    modelo = args.modelo or modelo_padrao
    os_cfg = _comum.config_opensearch()
    auth = (os_cfg["usuario"], os_cfg["senha"]) if os_cfg["usuario"] else None
    cliente, groq_modelo = _comum.groq_client()

    print("=" * 60)
    print("  CONTEXTUAL RETRIEVAL (#T09) - Aula 4")
    print("=" * 60)
    print(f"Indice: {args.indice} | Documentos: {args.limite} | LLM: {groq_modelo}")

    if args.recriar:
        apagar_indice(os_cfg["url"], args.indice, auth)

    store = OpenSearchDocumentStore(
        hosts=os_cfg["url"], index=args.indice, embedding_dim=_comum.dimensao_do_modelo(modelo),
        http_auth=auth, use_ssl=False, verify_certs=False,
    )

    docs_corpus = _comum.carregar_corpus(limite=args.limite)
    documentos = []
    print(f"\nGerando contexto com a Groq para {len(docs_corpus)} documentos...")
    for n, d in enumerate(docs_corpus, 1):
        try:
            contexto = gerar_contexto(cliente, groq_modelo, d.get("titulo", ""), d["texto"])
        except Exception as e:
            contexto = ""
            if n == 1:
                print(f"  (aviso: falha ao gerar contexto - {e})")
        enriquecido = f"Contexto: {contexto}\n\n{d['texto']}" if contexto else d["texto"]
        documentos.append(Document(
            content=enriquecido,
            meta={"id_original": d["id"], "titulo": d.get("titulo", ""), "contexto": contexto},
        ))
        if n <= 2:
            print(f"\n--- Exemplo {n} ---")
            print(f"  ANTES : {d['texto'][:120]}...")
            print(f"  DEPOIS: Contexto: {contexto[:120]}...")
        if n % 20 == 0:
            print(f"  ...{n} documentos processados")

    print(f"\nGerando embeddings de {len(documentos)} documentos enriquecidos...")
    embedder = OllamaDocumentEmbedder(model=modelo, url=base_url)
    docs_com_vetor = embedder.run(documents=documentos)["documents"]

    print("Gravando no OpenSearch...")
    qtd = store.write_documents(docs_com_vetor, policy=DuplicatePolicy.OVERWRITE)
    print(f"\nPronto! {qtd} documentos contextualizados no indice '{args.indice}'.")
    print(f"Compare a busca: python 02_busca_hibrida_rrf.py --indice {args.indice} --query \"...\"")


if __name__ == "__main__":
    main()
