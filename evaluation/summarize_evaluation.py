#!/usr/bin/env python3
"""
Build consolidated human-evaluation metrics from annotator spreadsheets.

Outputs a workbook with two sheets:
- RESUMO_EXECUTIVO: summary metrics by persona, criterio, and total
- BASE_DE_CALCULO: row-level values used for all summary calculations

Usage:
    python resumo_avaliacao_humana.py
    python resumo_avaliacao_humana.py --input-dir /path/to/avaliacaofinalhumano
    python resumo_avaliacao_humana.py --spreadsheet Planilha-Avaliação-Paula-AIMHEALTH.xlsx
    python resumo_avaliacao_humana.py --mask-spreadsheet /path/to/mascara_reordenacao.xlsx
"""

import argparse
import glob
import os
import re
import sys
import unicodedata

import pandas as pd


NOTA_MAP = {"Nota 1": 1, "Nota 2": 2, "Nota 3": 3}
CRITERIOS = ("empatia", "personalizacao", "adequacao")
VERSOES = (1, 2, 3)
# Tie-break requested by user: V3 > V2 > V1
VERSAO_PRIORIDADE = {"versao_1": 1, "versao_2": 2, "versao_3": 3}
SKIP_SHEETS = {"avaliacao_humana", "resumo_executivo", "base_de_calculo"}
MASK_METRICS = ("empatia", "personalizacao", "adequacao")
MASK_REQUIRED_COLUMNS = {"id", "de", "para"}


def normalize_text(value):
    """Lowercase and strip accents/spaces for robust column matching."""
    txt = "" if value is None else str(value)
    txt = unicodedata.normalize("NFKD", txt).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"\s+", "", txt).lower()


def convert_nota(value):
    """Convert 'Nota X' to int, pass through numerics, else None."""
    if pd.isna(value):
        return None
    if isinstance(value, str):
        mapped = NOTA_MAP.get(value.strip())
        if mapped is not None:
            return mapped
    try:
        return int(value)
    except (ValueError, TypeError):
        return None


def score_to_response_idx(token):
    """Translate suffix token into response index, handling pandas duplicates like 1.1."""
    idx = int(float(token))
    if "." in token:
        idx += 1
    return idx


def find_score_columns(columns):
    """
    Detect criteria columns for each response version.

    Returns:
      {
        1: {"empatia": "Empatia1", "personalizacao": "Personalização1", "adequacao": "Adequação1"},
        2: {...},
        3: {...}
      }
    """
    result = {1: {}, 2: {}, 3: {}}

    for original_col in columns:
        col = normalize_text(original_col)

        m_emp = re.match(r"empatia(\d+(?:\.\d+)?)$", col)
        if m_emp:
            resp_idx = score_to_response_idx(m_emp.group(1))
            if resp_idx in result:
                result[resp_idx]["empatia"] = original_col
            continue

        m_pers = re.match(r"personalizacao(\d+(?:\.\d+)?)$", col)
        if m_pers:
            resp_idx = score_to_response_idx(m_pers.group(1))
            if resp_idx in result:
                result[resp_idx]["personalizacao"] = original_col
            continue

        m_adeq = re.match(r"(?:adequacao|assertividade)(\d+(?:\.\d+)?)$", col)
        if m_adeq:
            resp_idx = score_to_response_idx(m_adeq.group(1))
            if resp_idx in result:
                result[resp_idx]["adequacao"] = original_col

    filtered = {}
    for versao, cols in result.items():
        if all(c in cols for c in CRITERIOS):
            filtered[versao] = cols
    return filtered


def calc_nota_final(values):
    """Arithmetic mean of three criterion scores, or None if any is missing."""
    if any(v is None for v in values):
        return None
    return round(sum(values) / 3.0, 4)


def choose_winner(scores_by_version):
    """Pick a single winner using max score and tie-break V3 > V2 > V1."""
    valid = {}
    for version, score in scores_by_version.items():
        if score is None:
            continue
        if pd.isna(score):
            continue
        valid[version] = float(score)

    if not valid:
        return None

    max_score = max(valid.values())
    winners = [v for v, s in valid.items() if s == max_score]
    winners.sort(key=lambda v: VERSAO_PRIORIDADE[v], reverse=True)
    return winners[0]


def extract_avaliador(filepath):
    """Extract evaluator name from file pattern Planilha-Avaliação-<Nome>-AIMHEALTH.xlsx."""
    name = os.path.basename(filepath)
    match = re.search(r"Planilha-Avalia..o-(.*?)-AIMHEALTH\.xlsx", name, flags=re.IGNORECASE)
    return match.group(1) if match else "desconhecido"


def parse_mask_target(value):
    """Parse mask target like 'personalizacao3' and return canonical column name."""
    token = normalize_text(value)
    match = re.match(r"(empatia|personalizacao|adequacao|notafinal)([123])$", token)
    if not match:
        return None
    criterio, versao = match.groups()
    return f"{criterio}{versao}"


def _safe_int(value):
    """Convert value to int for id matching, returning None on failure."""
    if value is None or pd.isna(value):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def recompute_row_scores(row):
    """Recompute notafinal and winner columns after mask edits on one row."""
    for versao in VERSOES:
        row[f"notafinal{versao}"] = calc_nota_final(
            [
                row.get(f"empatia{versao}"),
                row.get(f"personalizacao{versao}"),
                row.get(f"adequacao{versao}"),
            ]
        )

    row["winner_nota_final"] = choose_winner(
        {
            "versao_1": row.get("notafinal1"),
            "versao_2": row.get("notafinal2"),
            "versao_3": row.get("notafinal3"),
        }
    )
    row["winner_empatia"] = choose_winner(
        {
            "versao_1": row.get("empatia1"),
            "versao_2": row.get("empatia2"),
            "versao_3": row.get("empatia3"),
        }
    )
    row["winner_personalizacao"] = choose_winner(
        {
            "versao_1": row.get("personalizacao1"),
            "versao_2": row.get("personalizacao2"),
            "versao_3": row.get("personalizacao3"),
        }
    )
    row["winner_adequacao"] = choose_winner(
        {
            "versao_1": row.get("adequacao1"),
            "versao_2": row.get("adequacao2"),
            "versao_3": row.get("adequacao3"),
        }
    )
    row["melhor_saida_media"] = row["winner_nota_final"]
    return row


def apply_mask(base_df, mask_spreadsheet, mask_sheet=None):
    """
    Apply mask operations loaded from spreadsheet.

    Expected columns in mask sheet:
      - id (required)
      - de (required): source column, e.g. personalizacao3
      - para (required): destination column, e.g. personalizacao2
    Optional filters:
      - persona
      - avaliador
      - arquivo
    """
    if not os.path.exists(mask_spreadsheet):
        raise FileNotFoundError(f"Mask spreadsheet not found: {mask_spreadsheet}")

    mask_df = pd.read_excel(mask_spreadsheet, sheet_name=mask_sheet, engine="openpyxl")
    if mask_df.empty:
        print(f"Warning: mask sheet is empty: {mask_spreadsheet}")
        return base_df

    rename_map = {col: normalize_text(col) for col in mask_df.columns}
    mask_df = mask_df.rename(columns=rename_map)
    missing = [c for c in MASK_REQUIRED_COLUMNS if c not in mask_df.columns]
    if missing:
        raise ValueError(
            "Mask spreadsheet missing required columns: "
            f"{missing}. Required columns are: {sorted(MASK_REQUIRED_COLUMNS)}"
        )

    updated_rows = set()
    applied_rules = 0
    for i, rule in mask_df.iterrows():
        row_id = _safe_int(rule.get("id"))
        src_col = parse_mask_target(rule.get("de"))
        dst_col = parse_mask_target(rule.get("para"))

        if row_id is None or src_col is None or dst_col is None:
            print(
                f"Warning: skipping mask row {i + 2} due to invalid id/de/para values "
                f"(id={rule.get('id')}, de={rule.get('de')}, para={rule.get('para')})."
            )
            continue

        selector = base_df["id"].apply(_safe_int) == row_id

        if "persona" in mask_df.columns and not pd.isna(rule.get("persona")):
            selector &= (base_df["persona"].astype(str) == str(rule.get("persona")))
        if "avaliador" in mask_df.columns and not pd.isna(rule.get("avaliador")):
            selector &= (base_df["avaliador"].astype(str) == str(rule.get("avaliador")))
        if "arquivo" in mask_df.columns and not pd.isna(rule.get("arquivo")):
            selector &= (base_df["arquivo"].astype(str) == str(rule.get("arquivo")))

        matched_idx = base_df.index[selector].tolist()
        if not matched_idx:
            print(
                f"Warning: mask row {i + 2} did not match any record "
                f"(id={row_id}, de={src_col}, para={dst_col})."
            )
            continue

        base_df.loc[selector, dst_col] = base_df.loc[selector, src_col]
        updated_rows.update(matched_idx)
        applied_rules += 1

    if updated_rows:
        base_df.loc[list(updated_rows)] = base_df.loc[list(updated_rows)].apply(recompute_row_scores, axis=1)

    print(f"Mask rules applied: {applied_rules}")
    print(f"Rows updated by mask: {len(updated_rows)}")
    return base_df


def process_spreadsheet(filepath):
    """Read all persona sheets from one spreadsheet and return row-level DataFrame."""
    xl = pd.ExcelFile(filepath, engine="openpyxl")
    rows = []
    avaliador = extract_avaliador(filepath)
    source_file = os.path.basename(filepath)

    for sheet_name in xl.sheet_names:
        if normalize_text(sheet_name) in SKIP_SHEETS:
            continue

        df = xl.parse(sheet_name)
        if df.empty:
            continue

        score_cols = find_score_columns(df.columns)
        if len(score_cols) < 3:
            print(f"Warning: skipping sheet '{sheet_name}' in '{source_file}' (missing score columns)")
            continue

        for cols in score_cols.values():
            for col in cols.values():
                df[col] = df[col].apply(convert_nota)

        for _, row in df.iterrows():
            empatia = {}
            personalizacao = {}
            adequacao = {}
            notafinal = {}

            for versao in VERSOES:
                cols = score_cols[versao]
                empatia[versao] = row.get(cols["empatia"])
                personalizacao[versao] = row.get(cols["personalizacao"])
                adequacao[versao] = row.get(cols["adequacao"])
                notafinal[versao] = calc_nota_final(
                    [empatia[versao], personalizacao[versao], adequacao[versao]]
                )

            winner_nota_final = choose_winner(
                {
                    "versao_1": notafinal[1],
                    "versao_2": notafinal[2],
                    "versao_3": notafinal[3],
                }
            )
            winner_empatia = choose_winner(
                {"versao_1": empatia[1], "versao_2": empatia[2], "versao_3": empatia[3]}
            )
            winner_personalizacao = choose_winner(
                {
                    "versao_1": personalizacao[1],
                    "versao_2": personalizacao[2],
                    "versao_3": personalizacao[3],
                }
            )
            winner_adequacao = choose_winner(
                {"versao_1": adequacao[1], "versao_2": adequacao[2], "versao_3": adequacao[3]}
            )

            rows.append(
                {
                    "arquivo": source_file,
                    "avaliador": avaliador,
                    "persona": sheet_name,
                    "id": row.get("id"),
                    "dataset": row.get("dataset"),
                    "input": row.get("input"),
                    "empatia1": empatia[1],
                    "personalizacao1": personalizacao[1],
                    "adequacao1": adequacao[1],
                    "notafinal1": notafinal[1],
                    "empatia2": empatia[2],
                    "personalizacao2": personalizacao[2],
                    "adequacao2": adequacao[2],
                    "notafinal2": notafinal[2],
                    "empatia3": empatia[3],
                    "personalizacao3": personalizacao[3],
                    "adequacao3": adequacao[3],
                    "notafinal3": notafinal[3],
                    "melhor_saida_media": winner_nota_final,
                    "winner_empatia": winner_empatia,
                    "winner_personalizacao": winner_personalizacao,
                    "winner_adequacao": winner_adequacao,
                    "winner_nota_final": winner_nota_final,
                }
            )

    return pd.DataFrame(rows)


def collect_spreadsheets(input_dir, single_spreadsheet=None):
    """Collect spreadsheets from one file or from folder pattern."""
    if single_spreadsheet:
        if not os.path.exists(single_spreadsheet):
            raise FileNotFoundError(f"Spreadsheet not found: {single_spreadsheet}")
        return [single_spreadsheet]

    pattern = os.path.join(input_dir, "Planilha-Avaliação-*-AIMHEALTH.xlsx")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No spreadsheets found with pattern: {pattern}")
    return files


def best_version_from_values(v1, v2, v3):
    """Determine best version from three numeric values with tie-break rule."""
    return choose_winner({"versao_1": v1, "versao_2": v2, "versao_3": v3})


def wins_count(series):
    """Count wins from a winner series, guaranteeing all three versions are present."""
    counts = series.value_counts()
    return {
        "versao_1": int(counts.get("versao_1", 0)),
        "versao_2": int(counts.get("versao_2", 0)),
        "versao_3": int(counts.get("versao_3", 0)),
    }


def append_summary_row(rows, categoria, subcategoria, metrica, v1, v2, v3):
    """Append one standardized summary row."""
    rows.append(
        {
            "categoria": categoria,
            "subcategoria": subcategoria,
            "metrica": metrica,
            "versao_1": v1,
            "versao_2": v2,
            "versao_3": v3,
            "melhor_versao": best_version_from_values(v1, v2, v3),
        }
    )


def build_summary_table(base_df):
    """Build summary DataFrame with metrics by persona, criterio, and total."""
    rows = []

    # Total
    append_summary_row(
        rows,
        "Total",
        "Geral",
        "media_nota_final",
        base_df["notafinal1"].mean(),
        base_df["notafinal2"].mean(),
        base_df["notafinal3"].mean(),
    )
    total_wins = wins_count(base_df["winner_nota_final"].dropna())
    append_summary_row(
        rows,
        "Total",
        "Geral",
        "vitorias_nota_final",
        total_wins["versao_1"],
        total_wins["versao_2"],
        total_wins["versao_3"],
    )

    # Por persona
    for persona, group in base_df.groupby("persona", dropna=False):
        append_summary_row(
            rows,
            "Por Persona",
            str(persona),
            "media_nota_final",
            group["notafinal1"].mean(),
            group["notafinal2"].mean(),
            group["notafinal3"].mean(),
        )
        persona_wins = wins_count(group["winner_nota_final"].dropna())
        append_summary_row(
            rows,
            "Por Persona",
            str(persona),
            "vitorias_nota_final",
            persona_wins["versao_1"],
            persona_wins["versao_2"],
            persona_wins["versao_3"],
        )

    # Por criterio
    criterio_meta = {
        "Empatia": ("empatia1", "empatia2", "empatia3", "winner_empatia"),
        "Personalizacao": (
            "personalizacao1",
            "personalizacao2",
            "personalizacao3",
            "winner_personalizacao",
        ),
        "Adequacao": ("adequacao1", "adequacao2", "adequacao3", "winner_adequacao"),
    }
    for criterio_nome, (c1, c2, c3, winner_col) in criterio_meta.items():
        append_summary_row(
            rows,
            "Por Criterio",
            criterio_nome,
            "media_criterio",
            base_df[c1].mean(),
            base_df[c2].mean(),
            base_df[c3].mean(),
        )
        criterio_wins = wins_count(base_df[winner_col].dropna())
        append_summary_row(
            rows,
            "Por Criterio",
            criterio_nome,
            "vitorias_criterio",
            criterio_wins["versao_1"],
            criterio_wins["versao_2"],
            criterio_wins["versao_3"],
        )

    summary_df = pd.DataFrame(rows)
    return summary_df


def write_output(output_path, summary_df, base_df):
    """Write final workbook with the two requested tabs."""
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="RESUMO_EXECUTIVO", index=False)
        base_df.to_excel(writer, sheet_name="BASE_DE_CALCULO", index=False)


def build_base_dataframe(spreadsheets):
    """Process all spreadsheets and return one consolidated row-level DataFrame."""
    all_data = []
    for path in spreadsheets:
        df = process_spreadsheet(path)
        if not df.empty:
            all_data.append(df)

    if not all_data:
        return pd.DataFrame()
    return pd.concat(all_data, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description="Consolidate human evaluation metrics.")
    parser.add_argument(
        "--input-dir",
        default=os.path.dirname(os.path.abspath(__file__)),
        help="Directory containing Planilha-Avaliação-*-AIMHEALTH.xlsx files",
    )
    parser.add_argument(
        "--spreadsheet",
        help="Single spreadsheet path (optional). If provided, ignores --input-dir pattern search.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output workbook path. Default: <input-dir>/resumo_metricas_avaliacao_humana.xlsx",
    )
    parser.add_argument(
        "--mask-spreadsheet",
        default=None,
        help="Optional mask workbook used to remap scores by row id before summary.",
    )
    parser.add_argument(
        "--mask-sheet",
        default=0,
        help="Mask sheet name or index (default: 0, first sheet).",
    )
    args = parser.parse_args()

    output_path = args.output or os.path.join(args.input_dir, "resumo_metricas_avaliacao_humana.xlsx")

    try:
        spreadsheets = collect_spreadsheets(args.input_dir, args.spreadsheet)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)

    base_df = build_base_dataframe(spreadsheets)
    if base_df.empty:
        print("No valid data found in provided spreadsheets.", file=sys.stderr)
        sys.exit(1)

    if args.mask_spreadsheet:
        try:
            base_df = apply_mask(base_df, args.mask_spreadsheet, args.mask_sheet)
        except (FileNotFoundError, ValueError) as exc:
            print(str(exc), file=sys.stderr)
            sys.exit(1)

    summary_df = build_summary_table(base_df)
    write_output(output_path, summary_df, base_df)

    print(f"Processed spreadsheets: {len(spreadsheets)}")
    print(f"Total evaluated rows: {len(base_df)}")
    print(f"Personas: {sorted(base_df['persona'].dropna().unique().tolist())}")
    print(f"Output file: {output_path}")


if __name__ == "__main__":
    main()
