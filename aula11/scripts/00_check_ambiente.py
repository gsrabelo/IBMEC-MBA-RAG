"""
00_check_ambiente.py - Confere o ambiente da Aula 11 (tecnicas complementares).

Obrigatorios: OpenSearch, Ollama, Groq, corpus.
Opcionais (por tecnica): llmlingua (03), ragatouille+torch (04), sentence-transformers
+Pillow (05/CLIP), dspy (06). Cada um so e necessario para o seu script.

Uso:
    python 00_check_ambiente.py
    python 00_check_ambiente.py --testar-groq
"""

import argparse
import importlib.util

import _comum

_comum.carregar_env()


def ok(b):
    return "OK" if b else "FALHOU"


def tem(mod):
    return importlib.util.find_spec(mod) is not None


def checar_opensearch():
    try:
        n = _comum.abrir_store().count_documents()
        print(f"[OpenSearch] {ok(True)} - indice '{_comum.INDICE}' com {n} documentos")
        if n == 0:
            print("             (vazio) rode: python 01_indexar.py")
        return True
    except Exception as e:
        print(f"[OpenSearch] {ok(False)} - {e}")
        return False


def checar_ollama():
    try:
        emb = _comum.text_embedder()
        if hasattr(emb, "warm_up"):
            emb.warm_up()
        v = emb.run(text="teste")["embedding"]
        print(f"[Ollama]     {ok(True)} - embedding com {len(v)} dimensoes")
        return True
    except Exception as e:
        print(f"[Ollama]     {ok(False)} - {e}")
        return False


def checar_groq(testar):
    api_key, modelo, _ = _comum.config_groq()
    if not api_key:
        print(f"[Groq]       {ok(False)} - GROQ_API_KEY ausente")
        return False
    if not testar:
        print(f"[Groq]       {ok(True)} - chave presente (modelo {modelo})")
        return True
    try:
        cliente, modelo = _comum.groq_client()
        r = _comum.gerar_texto(cliente, modelo, "Responda apenas: ok", max_tokens=5)
        print(f"[Groq]       {ok(True)} - resposta: {r!r}")
        return True
    except Exception as e:
        print(f"[Groq]       {ok(False)} - {e}")
        return False


def checar_corpus():
    try:
        docs = _comum.carregar_corpus()
        print(f"[Corpus]     {ok(True)} - {len(docs)} documentos, {len(_comum.carregar_queries())} queries")
        return True
    except Exception as e:
        print(f"[Corpus]     {ok(False)} - {e}")
        return False


def checar_opcionais():
    print("-" * 60)
    print("Libs OPCIONAIS (cada uma so e necessaria para o seu script):")
    print(f"  [03 Compressao] llmlingua          : {'OK' if tem('llmlingua') else 'faltando -> pip install llmlingua'}")
    print(f"  [04 ColBERT]    pylate             : {'OK' if tem('pylate') else 'faltando -> pip install pylate'}")
    print(f"  [05 Multimodal] sentence-transformers+PIL+pymupdf : {'OK' if tem('sentence_transformers') and tem('PIL') and (tem('fitz') or tem('pymupdf')) else 'faltando -> pip install sentence-transformers pillow pymupdf docling'}")
    print(f"  [05 ColPali]    byaldi             : {'OK' if tem('byaldi') else 'faltando -> pip install byaldi (pesado/GPU; ver README)'}")
    print(f"  [06 DSPy]       dspy               : {'OK' if tem('dspy') else 'faltando -> pip install dspy-ai'}")


def main():
    parser = argparse.ArgumentParser(description="Checagem de ambiente da Aula 11.")
    parser.add_argument("--testar-groq", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("  CHECAGEM DE AMBIENTE - Aula 11 (Tecnicas Complementares)")
    print("=" * 60)
    obrig = [checar_opensearch(), checar_ollama(), checar_groq(args.testar_groq), checar_corpus()]
    checar_opcionais()
    print("-" * 60)
    print("Base pronta." if all(obrig) else "Resolva os itens FALHOU (obrigatorios).")


if __name__ == "__main__":
    main()
