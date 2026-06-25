"""
01_preparar_dados.py - Prepara os dados da Aula 10.

(1) Cria o banco SQLite 'juridico_segpub.db' com 4 tabelas e dados de exemplo
    (acordaos, ocorrencias, legislacao, doutrina) - usado pela ferramenta consultar_banco.
(2) Verifica o indice de jurisprudencia no OpenSearch (reusa o 'aula4_hibrido' da Aula 4),
    usado pela ferramenta buscar_documentos.

Uso:
    python 01_preparar_dados.py
    python 01_preparar_dados.py --recriar   # recria o SQLite do zero
"""

import argparse
import sqlite3

import _comum

_comum.carregar_env()

ACORDAOS = [
    ("AC 2344/2026", "TCU", "2026-04-10", "Min. Relator A", "Representacao", "Operacao de credito com garantia da Uniao; indicios de irregularidade.", "Procedente", 0.0, "Controle Externo"),
    ("AC 1163/2026", "TCU", "2026-03-05", "Min. Relator B", "Tomada de Contas", "Multa por contratacao emergencial sem justificativa de urgencia.", "Irregular", 0.0, "Controle Externo"),
    ("HC 127483", "STF", "2015-08-27", "Min. Dias Toffoli", "Habeas Corpus", "Colaboracao premiada; legitimidade do MPF para firmar o acordo.", "Concedido em parte", 0.0, "Penal"),
    ("AP 470", "STF", "2012-12-17", "Min. Joaquim Barbosa", "Acao Penal", "Esquema de desvio de recursos publicos; condenacao por corrupcao.", "Condenatorio", 7.5, "Penal"),
    ("REsp 1657156", "STJ", "2018-04-25", "Min. Relator C", "Recurso Especial", "Fornecimento de medicamentos; criterios para concessao.", "Provido", 0.0, "Administrativo"),
    ("AC 0987/2025", "TCU", "2025-11-20", "Min. Relator D", "Representacao", "Fraude em medicoes de obra publica; debito imputado.", "Irregular", 0.0, "Controle Externo"),
    ("HC 345678", "STF", "2023-06-15", "Min. Relator E", "Habeas Corpus", "Prisao preventiva em crime financeiro; requisitos do art. 312 CPP.", "Denegado", 0.0, "Penal"),
    ("AC 2210/2026", "TCU", "2026-02-18", "Min. Relator F", "Aposentadoria", "Registro de aposentadoria; acumulacao indevida de cargos.", "Ilegal", 0.0, "Administrativo"),
]

OCORRENCIAS = [
    ("BO-2024-0001", "2024-01-12", "Estelionato", "Sao Paulo", "SP", "Em investigacao", 150000.0, "Fraude bancaria por engenharia social."),
    ("BO-2024-0002", "2024-02-03", "Lavagem de dinheiro", "Rio de Janeiro", "RJ", "Indiciado", 2500000.0, "Triangulacao de recursos via empresas de fachada."),
    ("BO-2024-0003", "2024-03-21", "Peculato", "Belo Horizonte", "MG", "Concluido", 800000.0, "Desvio de verba de programa social."),
    ("BO-2024-0004", "2024-05-09", "Estelionato", "Campinas", "SP", "Em investigacao", 42000.0, "Golpe do falso boleto."),
    ("BO-2024-0005", "2024-06-30", "Corrupcao", "Brasilia", "DF", "Indiciado", 5000000.0, "Pagamento de propina em licitacao."),
    ("BO-2024-0006", "2024-07-18", "Trafico de influencia", "Salvador", "BA", "Arquivado", 0.0, "Suposta intermediacao junto a orgao publico."),
    ("BO-2024-0007", "2024-08-22", "Estelionato", "Santos", "SP", "Concluido", 95000.0, "Investimento fraudulento (piramide)."),
]

LEGISLACAO = [
    ("Lei 12.850/2013", "Lei Federal", "2013-08-02", "Define organizacao criminosa e meios de obtencao de prova.", "Art. 1; Art. 2; Art. 3", "Vigente"),
    ("Lei 9.613/1998", "Lei Federal", "1998-03-03", "Crimes de lavagem de dinheiro; COAF.", "Art. 1; Art. 9", "Vigente"),
    ("Lei 8.666/1993", "Lei Federal", "1993-06-21", "Licitacoes e contratos (parcialmente revogada).", "Art. 24; Art. 25", "Revogada parcialmente"),
    ("Lei 14.133/2021", "Lei Federal", "2021-04-01", "Nova Lei de Licitacoes e Contratos.", "Art. 75; Art. 155", "Vigente"),
    ("Lei 11.340/2006", "Lei Federal", "2006-08-07", "Lei Maria da Penha; violencia domestica.", "Art. 5; Art. 7", "Vigente"),
    ("CPP - DL 3.689/1941", "Decreto-Lei", "1941-10-03", "Codigo de Processo Penal.", "Art. 312; Art. 240", "Vigente"),
]

DOUTRINA = [
    ("Colaboracao premiada: limites", "Autor X", "Revista de Direito Penal", 2022, "Analisa a natureza juridica do acordo de colaboracao.", "colaboracao premiada; processo penal"),
    ("Prisao preventiva e crimes financeiros", "Autora Y", "Revista de Ciencias Criminais", 2023, "Discute os requisitos do art. 312 do CPP.", "prisao preventiva; crime financeiro"),
    ("Improbidade e controle externo", "Autor Z", "Revista do TCU", 2024, "Relacao entre improbidade e julgamento de contas.", "improbidade; TCU; controle externo"),
    ("Lavagem de dinheiro: tipologias", "Autor W", "Revista de Direito Penal Economico", 2021, "Tipologias de lavagem e indicios.", "lavagem; COAF"),
]


def criar_sqlite(recriar):
    if recriar and _comum.DB_PATH.exists():
        _comum.DB_PATH.unlink()
    _comum.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(_comum.DB_PATH)
    c = con.cursor()
    c.executescript("""
        CREATE TABLE IF NOT EXISTS acordaos(id INTEGER PRIMARY KEY, numero TEXT UNIQUE, tribunal TEXT,
            data_julgamento TEXT, relator TEXT, tipo TEXT, ementa TEXT, resultado TEXT, pena_anos REAL, area_direito TEXT);
        CREATE TABLE IF NOT EXISTS ocorrencias(id INTEGER PRIMARY KEY, numero_bo TEXT UNIQUE, data TEXT,
            tipo_crime TEXT, municipio TEXT, estado TEXT, status TEXT, valor_envolvido_reais REAL, descricao TEXT);
        CREATE TABLE IF NOT EXISTS legislacao(id INTEGER PRIMARY KEY, numero TEXT UNIQUE, tipo TEXT,
            data_publicacao TEXT, ementa TEXT, artigos_principais TEXT, status_vigencia TEXT);
        CREATE TABLE IF NOT EXISTS doutrina(id INTEGER PRIMARY KEY, titulo TEXT, autores TEXT,
            publicacao TEXT, ano INTEGER, resumo TEXT, temas TEXT);
    """)
    c.executemany("INSERT OR REPLACE INTO acordaos(numero,tribunal,data_julgamento,relator,tipo,ementa,resultado,pena_anos,area_direito) VALUES (?,?,?,?,?,?,?,?,?)", ACORDAOS)
    c.executemany("INSERT OR REPLACE INTO ocorrencias(numero_bo,data,tipo_crime,municipio,estado,status,valor_envolvido_reais,descricao) VALUES (?,?,?,?,?,?,?,?)", OCORRENCIAS)
    c.executemany("INSERT OR REPLACE INTO legislacao(numero,tipo,data_publicacao,ementa,artigos_principais,status_vigencia) VALUES (?,?,?,?,?,?)", LEGISLACAO)
    c.executemany("INSERT OR REPLACE INTO doutrina(titulo,autores,publicacao,ano,resumo,temas) VALUES (?,?,?,?,?,?)", DOUTRINA)
    con.commit()
    contagem = {t: con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                for t in ["acordaos", "ocorrencias", "legislacao", "doutrina"]}
    con.close()
    print(f"SQLite pronto: {_comum.DB_PATH}")
    print(f"  registros: {contagem}")


def checar_indice():
    try:
        n = _comum.abrir_store(_comum.INDICE_TCU).count_documents()
        print(f"OpenSearch '{_comum.INDICE_TCU}': {n} documentos "
              + ("(ok)" if n > 0 else "(VAZIO - rode a Aula 4 p/ a ferramenta buscar_documentos)"))
    except Exception as e:
        print(f"OpenSearch: nao acessivel ({e}). A ferramenta buscar_documentos exige o indice da Aula 4.")


def main():
    parser = argparse.ArgumentParser(description="Prepara dados da Aula 10.")
    parser.add_argument("--recriar", action="store_true", help="recria o SQLite do zero")
    args = parser.parse_args()

    print("=" * 60)
    print("  PREPARACAO DE DADOS - Aula 10")
    print("=" * 60)
    criar_sqlite(args.recriar)
    checar_indice()


if __name__ == "__main__":
    main()
