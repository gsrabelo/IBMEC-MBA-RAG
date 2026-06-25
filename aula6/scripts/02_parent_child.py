"""
02_parent_child.py - Hierarchical Indexing / Parent-Child (#T07) com Haystack.

Problema: chunk grande perde precisao na busca; chunk pequeno perde contexto na
resposta. Solucao Parent-Child: quebrar o documento em niveis -
  - FILHOS (pequenos) -> usados na BUSCA (precisao)
  - PAIS (grandes)    -> usados na RESPOSTA (contexto)
Busca-se nos filhos; se varios filhos do mesmo pai aparecem, o AutoMergingRetriever
"sobe" para o pai inteiro (mais contexto para o LLM).

ARQUITETURA (opcao A - tudo persistido no OpenSearch, sem InMemory):
  - indice de BUSCA  ('<indice>')         -> SO as folhas, COM embedding (kNN)
  - indice da ARVORE ('<indice>_arvore')  -> TODOS os nos (pais + folhas), com os
                                             metadados de hierarquia. E o store que o
                                             AutoMergingRetriever consulta p/ achar o
                                             pai (ele busca por id: filter_documents
                                             {"field":"id","operator":"==",...}).
O pai NAO precisa de embedding (nunca e alvo de busca vetorial; so e buscado por id).

IDEMPOTENTE: a arvore e os embeddings sao calculados UMA vez. Em execucoes normais o
script so reabre os dois indices (sem re-split / re-embed). Use --recriar para refazer.

Inspecione como a lib estrutura pais/filhos no OpenSearch com --inspecionar.

Precisa de OpenSearch, Ollama e Groq.

Uso:
    python 02_parent_child.py --recriar
    python 02_parent_child.py --pergunta "o que e cadeia de custodia?"
    python 02_parent_child.py --inspecionar
    python 02_parent_child.py --pai 200 --filho 50 --top-k 6 --threshold 0.5
"""

import argparse

import requests
from haystack import Pipeline
from haystack.components.preprocessors import HierarchicalDocumentSplitter
from haystack.components.retrievers import AutoMergingRetriever
from haystack.document_stores.types import DuplicatePolicy
from haystack_integrations.components.retrievers.opensearch import (
    OpenSearchEmbeddingRetriever,
)

import _comum


def nome_indice_arvore(indice):
    return f"{indice}_arvore"


def apagar_indice(indice):
    os_cfg = _comum.config_opensearch()
    auth = (os_cfg["usuario"], os_cfg["senha"]) if os_cfg["usuario"] else None
    try:
        requests.delete(f"{os_cfg['url']}/{indice}", auth=auth, timeout=10)
    except Exception:
        pass


def contar(store):
    try:
        return store.count_documents()
    except Exception:
        return 0


def construir_arvore(store_folhas, store_arvore, pai, filho):
    """(Re)constroi a hierarquia e grava: folhas (com embedding) e arvore (todos os nos)."""
    # 1) Quebra cada documento em niveis (pais e filhos).
    splitter = HierarchicalDocumentSplitter(
        block_sizes={pai, filho}, split_overlap=0, split_by="word")
    todos_nos = splitter.run(documents=_comum.documentos_haystack())["documents"]
    folhas = [d for d in todos_nos if not d.meta.get("__children_ids")]
    pais = [d for d in todos_nos if d.meta.get("__children_ids")]

    # 2) Indexa a ARVORE INTEIRA (pais + folhas) no OpenSearch, SEM embedding.
    #    E daqui que o AutoMergingRetriever vai buscar o pai (por id).
    store_arvore.write_documents(todos_nos, policy=DuplicatePolicy.OVERWRITE)

    # 3) Indexa SO as folhas COM embedding no indice de BUSCA.
    embedder = _comum.doc_embedder()
    folhas_emb = embedder.run(documents=folhas)["documents"]
    store_folhas.write_documents(folhas_emb, policy=DuplicatePolicy.OVERWRITE)

    niveis = sorted({d.meta.get("__level") for d in todos_nos})
    print(f"  Arvore construida: {len(todos_nos)} nos | {len(pais)} pais | "
          f"{len(folhas)} folhas | niveis={niveis}")


def construir_responder(indice, pai=200, filho=50, top_k=6, threshold=0.5, recriar=False):
    """Abre/constroi os 2 indices e devolve responder(pergunta) usando auto-merging."""
    indice_arvore = nome_indice_arvore(indice)
    if recriar:
        apagar_indice(indice)
        apagar_indice(indice_arvore)

    store_folhas = _comum.abrir_store(indice)
    store_arvore = _comum.abrir_store(indice_arvore)

    # Idempotente: so recomputa se pedido ou se algum indice estiver vazio.
    precisa_construir = recriar or contar(store_folhas) == 0 or contar(store_arvore) == 0
    if precisa_construir:
        print("  Construindo a hierarquia (split + embeddings das folhas)...")
        construir_arvore(store_folhas, store_arvore, pai, filho)
    else:
        print(f"  Reaproveitando indices: '{indice}' ({contar(store_folhas)} folhas) e "
              f"'{indice_arvore}' ({contar(store_arvore)} nos).")

    # Pipeline de busca: embedding da query -> recupera filhos -> auto-merging (arvore).
    busca = Pipeline()
    busca.add_component("embedder", _comum.text_embedder())
    busca.add_component("retriever", OpenSearchEmbeddingRetriever(document_store=store_folhas, top_k=top_k))
    busca.add_component("merge", AutoMergingRetriever(document_store=store_arvore, threshold=threshold))
    busca.connect("embedder.embedding", "retriever.query_embedding")
    busca.connect("retriever.documents", "merge.documents")

    cliente, modelo = _comum.groq_client()

    def responder(pergunta):
        docs = busca.run({"embedder": {"text": pergunta}})["merge"]["documents"]
        contextos = [d.content for d in docs]
        resposta = _comum.responder_com_contexto(cliente, modelo, pergunta, contextos)
        return resposta, contextos

    return responder


def inspecionar(indice):
    """Mostra como pais e filhos ficam estruturados no indice da arvore (OpenSearch)."""
    indice_arvore = nome_indice_arvore(indice)
    store = _comum.abrir_store(indice_arvore)
    nos = store.filter_documents()
    if not nos:
        print(f"Indice '{indice_arvore}' vazio. Rode antes: python 02_parent_child.py --recriar")
        return

    por_nivel = {}
    for d in nos:
        por_nivel.setdefault(d.meta.get("__level"), []).append(d)

    print(f"Indice da arvore: '{indice_arvore}' | {len(nos)} nos no total")
    print("-" * 55)
    for nivel in sorted(por_nivel, key=lambda x: (x is None, x)):
        grupo = por_nivel[nivel]
        com_filhos = sum(1 for d in grupo if d.meta.get("__children_ids"))
        print(f"  level {nivel}: {len(grupo)} nos | {com_filhos} com filhos "
              f"| block_sizes={sorted({d.meta.get('__block_size') for d in grupo})}")

    pais = [d for d in nos if d.meta.get("__children_ids")]
    folhas = [d for d in nos if not d.meta.get("__children_ids")]
    if pais:
        p = pais[0]
        print("\nExemplo de PAI:")
        print(f"  id={p.id[:16]}... | level={p.meta.get('__level')} "
              f"| n_filhos={len(p.meta.get('__children_ids', []))}")
        print(f"  children_ids (3 primeiros): {[c[:12]+'...' for c in p.meta.get('__children_ids', [])[:3]]}")
    if folhas:
        f = folhas[0]
        print("\nExemplo de FILHO (folha):")
        print(f"  id={f.id[:16]}... | level={f.meta.get('__level')} "
              f"| block_size={f.meta.get('__block_size')} | children_ids={f.meta.get('__children_ids')}")
        print(f"  parent_id={str(f.meta.get('__parent_id'))[:16]}...  <- aponta para o PAI acima")
    print("\nResumo: no indice de BUSCA so existem as folhas (com embedding); a ARVORE "
          "(pais + folhas) fica aqui. O AutoMergingRetriever pega o parent_id da folha "
          "recuperada e busca o pai por id NESTE indice.")


def main():
    parser = argparse.ArgumentParser(description="Parent-Child / Hierarchical (#T07).")
    parser.add_argument("--indice", default="aula6_parent_child", help="indice de busca (folhas)")
    parser.add_argument("--pergunta", default=None, help="pergunta (senao roda exemplos do corpus)")
    parser.add_argument("--pai", type=int, default=200, help="tamanho do bloco PAI (palavras)")
    parser.add_argument("--filho", type=int, default=50, help="tamanho do bloco FILHO (palavras)")
    parser.add_argument("--top-k", type=int, default=6, help="quantos filhos buscar")
    parser.add_argument("--threshold", type=float, default=0.5, help="fracao de filhos p/ subir ao pai")
    parser.add_argument("--recriar", action="store_true", help="recria os dois indices")
    parser.add_argument("--inspecionar", action="store_true",
                        help="mostra a estrutura pai/filho no indice da arvore e sai")
    args = parser.parse_args()

    _comum.carregar_env()
    print("=" * 60)
    print("  PARENT-CHILD / HIERARCHICAL (#T07) - Aula 6")
    print("=" * 60)

    if args.inspecionar:
        inspecionar(args.indice)
        return

    print(f"Busca: '{args.indice}' | Arvore: '{nome_indice_arvore(args.indice)}'")
    print(f"pai={args.pai} filho={args.filho} top_k={args.top_k} threshold={args.threshold}")

    responder = construir_responder(
        args.indice, args.pai, args.filho, args.top_k, args.threshold, args.recriar)

    _, perguntas = _comum.carregar_corpus()
    alvos = [args.pergunta] if args.pergunta else [p["pergunta"] for p in perguntas][:3]
    for q in alvos:
        print("\n" + "-" * 55)
        print(f"Pergunta: {q}")
        resposta, contextos = responder(q)
        print(f"Trechos usados: {len(contextos)} (tamanhos: {[len(c) for c in contextos]})")
        print(f"Resposta: {resposta}")


if __name__ == "__main__":
    main()
