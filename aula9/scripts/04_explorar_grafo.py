"""
04_explorar_grafo.py - Inspeciona o GRAFO gerado (torna o Graph RAG tangivel).

Le o arquivo graph_chunk_entity_relation.graphml (NetworkX) que o LightRAG criou e
mostra a estrutura do grafo de conhecimento:
  - total de entidades (nos) e relacoes (arestas);
  - entidades mais conectadas (maior grau) = os "hubs" do acervo;
  - vizinhos de uma entidade especifica (--entidade);
  - exporta um subgrafo de uma entidade para JSON (--exportar).

Nao precisa de Groq/Ollama - le so o arquivo do grafo.

Uso:
    python 04_explorar_grafo.py
    python 04_explorar_grafo.py --entidade "Colaboracao Premiada"
    python 04_explorar_grafo.py --entidade "MPF" --exportar subgrafo.json
"""

import argparse
import json

import networkx as nx

import _comum

_comum.carregar_env()


def carregar_grafo():
    return _comum.ler_graphml()   # leitura robusta (limpa null bytes/lixo apos </graphml>)


def rotulo(g, n):
    """Nome legivel do no (LightRAG guarda o nome em atributos ou no proprio id)."""
    d = g.nodes[n]
    return d.get("entity_id") or d.get("entity_name") or d.get("description", "")[:40] or str(n)


def achar_no(g, termo):
    termo_l = termo.lower()
    for n in g.nodes:
        if termo_l in str(n).lower() or termo_l in rotulo(g, n).lower():
            return n
    return None


def main():
    parser = argparse.ArgumentParser(description="Explora o grafo de conhecimento (Aula 9).")
    parser.add_argument("--top", type=int, default=15, help="quantos hubs listar")
    parser.add_argument("--entidade", default=None, help="mostra vizinhos desta entidade")
    parser.add_argument("--exportar", default=None, help="salva o subgrafo da entidade em JSON")
    args = parser.parse_args()

    print("=" * 60)
    print("  EXPLORACAO DO GRAFO DE CONHECIMENTO - Aula 9")
    print("=" * 60)
    g = carregar_grafo()
    print(f"Entidades (nos): {g.number_of_nodes()} | Relacoes (arestas): {g.number_of_edges()}")

    print(f"\nTop {args.top} entidades mais conectadas (hubs):")
    graus = sorted(g.degree, key=lambda x: x[1], reverse=True)[: args.top]
    for n, grau in graus:
        print(f"  {grau:>3}  {rotulo(g, n)}")

    if args.entidade:
        n = achar_no(g, args.entidade)
        if not n:
            print(f"\n[!] Entidade '{args.entidade}' nao encontrada no grafo.")
            return
        print(f"\nVizinhos de '{rotulo(g, n)}':")
        vizinhos = list(g.neighbors(n))
        for v in vizinhos:
            aresta = g.get_edge_data(n, v) or {}
            desc = (aresta.get("description") or aresta.get("keywords") or "").strip()
            print(f"  -> {rotulo(g, v)}" + (f"   [{desc[:70]}]" if desc else ""))

        if args.exportar:
            sub = {
                "entidade": rotulo(g, n),
                "vizinhos": [
                    {"entidade": rotulo(g, v),
                     "relacao": (g.get_edge_data(n, v) or {}).get("description", "")}
                    for v in vizinhos
                ],
            }
            with open(args.exportar, "w", encoding="utf-8") as f:
                json.dump(sub, f, ensure_ascii=False, indent=2)
            print(f"\nSubgrafo exportado para {args.exportar} ({len(vizinhos)} vizinhos).")


if __name__ == "__main__":
    main()
