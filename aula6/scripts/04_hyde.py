"""
04_hyde.py - HyDE: Hypothetical Document Embeddings (#T05) com Haystack.

Problema (gap semantico): a pergunta do usuario ("policial pode revistar na rua?")
usa palavras diferentes dos documentos ("busca pessoal", "art. 240 CPP"), entao o
embedding da pergunta fica "longe" dos documentos certos.

Solucao HyDE: pedir ao LLM para escrever um documento HIPOTETICO que responderia a
pergunta (em linguagem juridica). Buscamos pelo embedding DESSE documento (mais
parecido com o corpus) em vez do embedding da pergunta crua.

Fluxo: pergunta -> LLM gera doc hipotetico -> embedding do doc -> busca no OpenSearch
-> resposta. (O doc hipotetico pode ter imprecisoes; ele serve SO para a busca.)

Precisa de OpenSearch, Ollama e Groq.

Uso:
    python 04_hyde.py --recriar
    python 04_hyde.py --pergunta "policial pode revistar alguem na rua?"
"""

import argparse

import requests
from haystack import Pipeline
from haystack.document_stores.types import DuplicatePolicy
from haystack_integrations.components.retrievers.opensearch import (
    OpenSearchEmbeddingRetriever,
)

import _comum

HYDE_PROMPT = (
    "Escreva um trecho de 80 a 120 palavras de um documento juridico brasileiro "
    "(lei, acordao ou doutrina) que responderia a pergunta abaixo. Use linguagem "
    "tecnica formal e cite artigos quando fizer sentido. Escreva APENAS o trecho.\n\n"
    "Pergunta: {pergunta}"
)


def construir_responder(indice, top_k=5, recriar=False):
    """Indexa o corpus (flat) e devolve responder(pergunta) usando HyDE."""
    os_cfg = _comum.config_opensearch()
    auth = (os_cfg["usuario"], os_cfg["senha"]) if os_cfg["usuario"] else None
    if recriar:
        try:
            requests.delete(f"{os_cfg['url']}/{indice}", auth=auth, timeout=10)
        except Exception:
            pass

    # Indexa os documentos (com embedding) no OpenSearch.
    store = _comum.abrir_store(indice)
    docs_emb = _comum.doc_embedder().run(documents=_comum.documentos_haystack())["documents"]
    store.write_documents(docs_emb, policy=DuplicatePolicy.OVERWRITE)

    busca = Pipeline()
    busca.add_component("embedder", _comum.text_embedder())
    busca.add_component("retriever", OpenSearchEmbeddingRetriever(document_store=store, top_k=top_k))
    busca.connect("embedder.embedding", "retriever.query_embedding")

    cliente, modelo = _comum.groq_client()

    def responder(pergunta):
        # 1) gerar o documento hipotetico
        hipotetico = _comum.gerar_texto(cliente, modelo, HYDE_PROMPT.format(pergunta=pergunta), max_tokens=250)
        # Se o LLM devolver vazio (modelo de reasoning, filtro, etc.), o embedder
        # receberia "" e o Ollama retornaria lista vazia -> "list index out of
        # range". Nesses casos, caimos para a pergunta crua na busca.
        texto_busca = hipotetico.strip() or pergunta
        if not hipotetico.strip():
            print("   [aviso] documento hipotetico veio vazio; buscando pela pergunta crua.")
        # 2) buscar pelo embedding do hipotetico (nao da pergunta crua)
        docs = busca.run({"embedder": {"text": texto_busca}})["retriever"]["documents"]
        contextos = [d.content for d in docs]
        # 3) responder com base nos trechos recuperados
        resposta = _comum.responder_com_contexto(cliente, modelo, pergunta, contextos)
        return resposta, contextos, hipotetico

    return responder


def main():
    parser = argparse.ArgumentParser(description="HyDE (#T05).")
    parser.add_argument("--indice", default="aula6_hyde", help="indice no OpenSearch")
    parser.add_argument("--pergunta", default=None, help="pergunta (senao roda exemplos do corpus)")
    parser.add_argument("--top-k", type=int, default=5, help="quantos trechos buscar")
    parser.add_argument("--recriar", action="store_true", help="recria o indice")
    args = parser.parse_args()

    _comum.carregar_env()
    print("=" * 60)
    print("  HyDE - Hypothetical Document Embeddings (#T05) - Aula 6")
    print("=" * 60)
    print(f"Indice: {args.indice} | top_k={args.top_k}")
    print("Indexando o corpus...")

    responder = construir_responder(args.indice, args.top_k, args.recriar)

    _, perguntas = _comum.carregar_corpus()
    alvos = [args.pergunta] if args.pergunta else [p["pergunta"] for p in perguntas][:3]
    for q in alvos:
        print("\n" + "-" * 55)
        print(f"Pergunta: {q}")
        resposta, contextos, hipotetico = responder(q)
        print(f"Documento hipotetico (usado na busca): {hipotetico[:160]}...")
        print(f"Trechos recuperados: {len(contextos)}")
        print(f"Resposta: {resposta}")


if __name__ == "__main__":
    main()
