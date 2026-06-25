"""
01_nlp_basico_ptbr.py - Fundamentos de NLP em portugues.

Mostra, passo a passo, 3 conceitos basicos antes dos embeddings:
  1. Tokenizacao  - quebrar o texto em frases e palavras
  2. Stemming     - reduzir a palavra ao seu radical (ex.: juridico -> jurid)
  3. Bag-of-Words - representar frases como contagem de palavras

Por padrao, a tokenizacao usa um documento do corpus juridico
(datasets/corpus_juridico_aula1.json). Use --texto para informar outro texto.

Usa apenas NLTK + scikit-learn (nao precisa de OpenSearch, Ollama nem Groq).

Uso:
    python 01_nlp_basico_ptbr.py
    python 01_nlp_basico_ptbr.py --doc 2
    python 01_nlp_basico_ptbr.py --texto "O juiz absolveu o reu por falta de provas."
"""

import argparse

import nltk
from nltk.stem import RSLPStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import CountVectorizer

import _comum


def garantir_dados_nltk():
    """Baixa os pacotes do NLTK necessarios (so baixa na primeira vez)."""
    for pacote in ["punkt", "punkt_tab", "rslp"]:
        try:
            nltk.data.find(pacote)
        except LookupError:
            nltk.download(pacote, quiet=True)


def demo_tokenizacao(texto):
    print("\n1) TOKENIZACAO")
    print("-" * 50)
    frases = sent_tokenize(texto, language="portuguese")
    palavras = word_tokenize(texto, language="portuguese")
    print(f"Texto (inicio): {texto[:120]}...")
    print(f"Total de frases:  {len(frases)}")
    print(f"Total de palavras: {len(palavras)}")
    print(f"Primeira frase: {frases[0] if frases else '(vazio)'}")
    print(f"Primeiras 15 palavras: {palavras[:15]}")


def demo_stemming():
    print("\n2) STEMMING (radical da palavra)")
    print("-" * 50)
    stemmer = RSLPStemmer()
    for palavra in ["justica", "juridico", "juiz", "condenado", "condenacao"]:
        print(f"  {palavra:14s} -> {stemmer.stem(palavra)}")


def demo_bag_of_words():
    print("\n3) BAG-OF-WORDS (contagem de palavras)")
    print("-" * 50)
    corpus = ["o juiz absolveu o reu", "o reu absolveu o juiz"]
    vectorizer = CountVectorizer()
    matriz = vectorizer.fit_transform(corpus)
    vocab = vectorizer.get_feature_names_out()
    print("Vocabulario:", list(vocab))
    for frase, linha in zip(corpus, matriz.toarray()):
        contagem = {palavra: int(qtd) for palavra, qtd in zip(vocab, linha)}
        print(f"  '{frase}' -> {contagem}")
    print("Repare: as duas frases tem a MESMA contagem, mas sentidos diferentes.")
    print("(esse e o limite do Bag-of-Words; embeddings resolvem isso)")


def texto_do_corpus(indice_doc):
    """Pega o texto (titulo + conteudo) de um documento do corpus pelo indice (1..N)."""
    corpus = _comum.carregar_corpus()
    indice = max(1, min(indice_doc, len(corpus)))  # garante que fica dentro do corpus
    doc = corpus[indice - 1]
    print(f"Usando o documento [{doc['id']}] do corpus: {doc['titulo']}")
    return doc["texto"]


def main():
    parser = argparse.ArgumentParser(description="Fundamentos de NLP em portugues.")
    parser.add_argument("--texto", default=None,
                        help="texto para tokenizar (padrao: um documento do corpus)")
    parser.add_argument("--doc", type=int, default=1,
                        help="qual documento do corpus usar quando nao houver --texto (padrao: 1)")
    args = parser.parse_args()

    print("=" * 60)
    print("  NLP BASICO EM PORTUGUES - Aula 1")
    print("=" * 60)
    garantir_dados_nltk()
    texto = args.texto if args.texto else texto_do_corpus(args.doc)
    demo_tokenizacao(texto)
    demo_stemming()
    demo_bag_of_words()


if __name__ == "__main__":
    main()
