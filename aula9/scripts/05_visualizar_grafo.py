"""
05_visualizar_grafo.py - Gera uma VISUALIZACAO INTERATIVA do grafo (HTML).

Le o graph_chunk_entity_relation.graphml (NetworkX) que o LightRAG criou e produz um
arquivo HTML autossuficiente (usa vis-network via CDN) que voce abre no navegador para
APRESENTAR o grafo: nos coloridos por tipo de entidade, tamanho por grau (hubs maiores),
arestas com a descricao da relacao no hover, fisica/zoom/arraste e busca.

Nao precisa de Groq/Ollama/servidor - le so o arquivo do grafo. Bom para sala de aula.

Uso:
    python 05_visualizar_grafo.py
    python 05_visualizar_grafo.py --saida grafo.html
    python 05_visualizar_grafo.py --entidade "Criptomoedas"   # foca a entidade + vizinhos
"""

import argparse
import json

import networkx as nx

import _comum

_comum.carregar_env()

# cores por tipo de entidade (vis-network)
CORES = {
    "organization": "#4e79a7", "person": "#e15759", "concept": "#59a14f",
    "operation": "#f28e2b", "method": "#b07aa1", "artifact": "#76b7b2",
    "content": "#edc948", "event": "#ff9da7", "data": "#9c755f", "location": "#bab0ac",
}
COR_PADRAO = "#bab0ac"

TEMPLATE = """<!DOCTYPE html>
<html lang="pt-br"><head><meta charset="utf-8">
<title>{titulo}</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/vis-network/9.1.6/standalone/umd/vis-network.min.js"></script>
<style>
  body{{font-family:Segoe UI,Arial,sans-serif;margin:0;background:#1e1e2e;color:#eee}}
  #cab{{padding:10px 16px;background:#181825;border-bottom:1px solid #313244}}
  #cab h1{{font-size:16px;margin:0}}
  #cab small{{color:#a6adc8}}
  #rede{{width:100%;height:calc(100vh - 96px);background:#1e1e2e}}
  #leg{{padding:6px 16px;background:#181825;font-size:12px;border-top:1px solid #313244}}
  .chip{{display:inline-block;margin:2px 8px 2px 0}}
  .pt{{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:4px;vertical-align:middle}}
</style></head>
<body>
  <div id="cab"><h1>{titulo}</h1>
    <small>{n_nos} entidades &middot; {n_arestas} relações &middot; arraste, dê zoom e passe o mouse nas arestas</small></div>
  <div id="rede"></div>
  <div id="leg">{legenda}</div>
<script>
  const nodes = new vis.DataSet({nodes});
  const edges = new vis.DataSet({edges});
  const container = document.getElementById('rede');
  const data = {{nodes, edges}};
  const options = {{
    nodes: {{shape:'dot', font:{{color:'#eee',size:14}}, borderWidth:1}},
    edges: {{color:{{color:'#6c7086',highlight:'#f38ba8'}}, smooth:{{type:'continuous'}},
             arrows:{{to:{{enabled:true,scaleFactor:0.5}}}}, font:{{color:'#a6adc8',size:10,strokeWidth:0}}}},
    physics: {{barnesHut:{{gravitationalConstant:-8000, springLength:120}}, stabilization:{{iterations:200}}}},
    interaction: {{hover:true, tooltipDelay:120, navigationButtons:true, keyboard:true}}
  }};
  new vis.Network(container, data, options);
</script>
</body></html>"""


def carregar_grafo():
    return _comum.ler_graphml()   # leitura robusta (limpa null bytes/lixo apos </graphml>)


def rotulo(g, n):
    return (g.nodes[n].get("entity_id") or str(n)).replace("_", " ")


def subgrafo_da_entidade(g, termo):
    termo_l = termo.lower()
    alvo = next((n for n in g.nodes if termo_l in str(n).lower()
                 or termo_l in rotulo(g, n).lower()), None)
    if alvo is None:
        return None
    viz = set(g.neighbors(alvo)) | {alvo}
    return g.subgraph(viz).copy()


def construir(g):
    graus = dict(g.degree)
    nodes = []
    for n in g.nodes:
        tipo = g.nodes[n].get("entity_type", "UNKNOWN")
        desc = g.nodes[n].get("description", "")
        nodes.append({
            "id": str(n),
            "label": rotulo(g, n),
            "title": f"{rotulo(g, n)}  [{tipo}]\n{desc}",
            "color": CORES.get(str(tipo).lower(), COR_PADRAO),
            "value": graus.get(n, 1) + 1,
        })
    edges = []
    for u, v in g.edges:
        a = g.edges[u, v]
        edges.append({
            "from": str(u), "to": str(v),
            "title": a.get("description", ""),
            "label": (a.get("keywords", "") or "").split(",")[0],
        })
    return nodes, edges


def legenda(g):
    tipos = sorted({str(g.nodes[n].get("entity_type", "UNKNOWN")).lower() for n in g.nodes})
    chips = []
    for t in tipos:
        cor = CORES.get(t, COR_PADRAO)
        chips.append(f'<span class="chip"><span class="pt" style="background:{cor}"></span>{t}</span>')
    return "".join(chips)


def main():
    parser = argparse.ArgumentParser(description="Visualizacao interativa do grafo (Aula 9).")
    parser.add_argument("--saida", default=str(_comum.PASTA_AULA9 / "grafo_juridico.html"))
    parser.add_argument("--entidade", default=None, help="foca uma entidade + vizinhos")
    args = parser.parse_args()

    print("=" * 60)
    print("  VISUALIZACAO INTERATIVA DO GRAFO - Aula 9")
    print("=" * 60)
    g = carregar_grafo()
    titulo = "Grafo de Conhecimento Juridico (LightRAG)"
    if args.entidade:
        sg = subgrafo_da_entidade(g, args.entidade)
        if sg is None:
            print(f"[!] Entidade '{args.entidade}' nao encontrada.")
            return
        g = sg
        titulo += f" - foco: {args.entidade}"

    nodes, edges = construir(g)
    html = TEMPLATE.format(
        titulo=titulo, n_nos=g.number_of_nodes(), n_arestas=g.number_of_edges(),
        legenda=legenda(g),
        nodes=json.dumps(nodes, ensure_ascii=False),
        edges=json.dumps(edges, ensure_ascii=False))
    with open(args.saida, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"OK - {g.number_of_nodes()} entidades, {g.number_of_edges()} relacoes")
    print(f"Arquivo: {args.saida}")
    print("Abra no navegador para apresentar (arraste, zoom, hover nas arestas).")


if __name__ == "__main__":
    main()
