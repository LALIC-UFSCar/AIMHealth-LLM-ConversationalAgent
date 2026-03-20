"""Scoring and summary utilities for evaluation results."""

from __future__ import annotations

import pandas as pd

VERSAO_PRIORIDADE = {"versao_1": 1, "versao_2": 2, "versao_3": 3}


def calc_nota_final(empathy: int, personalization: int, thematic_adequacy: int) -> float:
    return round((empathy + personalization + thematic_adequacy) / 3, 4)


def determine_melhor_saida(nf1: float, nf2: float, nf3: float) -> str:
    scores = {"versao_1": nf1, "versao_2": nf2, "versao_3": nf3}
    max_score = max(scores.values())
    winners = [name for name, score in scores.items() if score == max_score]
    return ", ".join(sorted(winners))


def choose_winner(scores_by_version: dict[str, float | None]) -> str | None:
    valid = {k: v for k, v in scores_by_version.items() if v is not None and not pd.isna(v)}
    if not valid:
        return None
    max_score = max(valid.values())
    winners = [v for v, s in valid.items() if s == max_score]
    winners.sort(key=lambda v: VERSAO_PRIORIDADE[v], reverse=True)
    return winners[0]


def wins_count(series: pd.Series) -> dict[str, int]:
    counts = series.value_counts()
    return {
        "versao_1": int(counts.get("versao_1", 0)),
        "versao_2": int(counts.get("versao_2", 0)),
        "versao_3": int(counts.get("versao_3", 0)),
    }


def build_summary_table(base_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []

    def append_row(categoria: str, subcategoria: str, metrica: str, v1: float, v2: float, v3: float) -> None:
        rows.append(
            {
                "categoria": categoria,
                "subcategoria": subcategoria,
                "metrica": metrica,
                "versao_1": v1,
                "versao_2": v2,
                "versao_3": v3,
                "melhor_versao": choose_winner({"versao_1": v1, "versao_2": v2, "versao_3": v3}),
            }
        )

    append_row(
        "Total",
        "Geral",
        "media_nota_final",
        base_df["notafinal1"].mean(),
        base_df["notafinal2"].mean(),
        base_df["notafinal3"].mean(),
    )

    total_wins = wins_count(base_df["winner_nota_final"].dropna())
    append_row("Total", "Geral", "vitorias_nota_final", total_wins["versao_1"], total_wins["versao_2"], total_wins["versao_3"])

    for persona, group in base_df.groupby("persona", dropna=False):
        append_row(
            "Por Persona",
            str(persona),
            "media_nota_final",
            group["notafinal1"].mean(),
            group["notafinal2"].mean(),
            group["notafinal3"].mean(),
        )
        persona_wins = wins_count(group["winner_nota_final"].dropna())
        append_row("Por Persona", str(persona), "vitorias_nota_final", persona_wins["versao_1"], persona_wins["versao_2"], persona_wins["versao_3"])

    return pd.DataFrame(rows)
