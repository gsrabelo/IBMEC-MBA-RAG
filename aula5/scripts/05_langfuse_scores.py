"""
05_langfuse_scores.py - Envia os scores de avaliacao para o LangFuse (Scores API).

Em producao, a gente registra as notas de qualidade (RAGAS) no LangFuse para
acompanhar a evolucao ao longo do tempo. Este script le os resultados salvos
pelo 02 e envia cada metrica como um "score" via a Scores API do LangFuse.

Precisa do LangFuse configurado no .env e dos resultados do 02 (ragas_resultados.json).

Uso:
    python 05_langfuse_scores.py
"""

import argparse
import json
import os

import _comum


def main():
    argparse.ArgumentParser(description="Envia scores RAGAS ao LangFuse.").parse_args()

    _comum.carregar_env()
    print("=" * 60)
    print("  ENVIAR SCORES AO LANGFUSE - Aula 5")
    print("=" * 60)

    if not _comum.langfuse_configurado():
        print("LangFuse nao configurado (.env sem LANGFUSE_PUBLIC_KEY/SECRET_KEY).")
        print("Veja GUIA_LANGFUSE_WINDOWS.md da Aula 3.")
        return
    if not _comum.RESULTADOS_RAGAS.exists():
        print(f"{_comum.RESULTADOS_RAGAS.name} nao encontrado. Rode primeiro: python 03_ragas_avaliar.py")
        return

    with open(_comum.RESULTADOS_RAGAS, "r", encoding="utf-8") as f:
        resultados = json.load(f)
    medias = resultados.get("medias", {})
    print(f"Metricas a enviar: {list(medias.keys())}")

    from langfuse import Langfuse

    host = os.getenv("LANGFUSE_HOST", "http://localhost:3000")
    lf = Langfuse(
        public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
        secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
        host=host,
    )

    # Cria um trace (via um evento) e anexa os scores a ele.
    trace_id = lf.create_trace_id()
    lf.create_event(
        trace_context={"trace_id": trace_id},
        name="avaliacao-ragas-aula5",
        input={"n_pares": resultados.get("n_pares")},
        output=medias,
    )
    for nome, valor in medias.items():
        lf.create_score(
            name=f"ragas_{nome}", value=float(valor), trace_id=trace_id,
            data_type="NUMERIC", comment="Media RAGAS - Aula 5",
        )
        print(f"  enviado: ragas_{nome} = {valor:.3f}")
    lf.flush()

    try:
        url = lf.get_trace_url(trace_id=trace_id)
    except Exception:
        url = f"{host} (procure o trace 'avaliacao-ragas-aula5')"
    print(f"\nPronto! Scores no LangFuse. Trace: {url}")


if __name__ == "__main__":
    main()
