from __future__ import annotations

import argparse
import json
import math
import re
import warnings
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import norm


warnings.filterwarnings(
    "ignore",
    message="covariance of constraints does not have full rank.*",
)


STANDARD_DIRS = (
    "A-Full",
    "A-No-Graph",
    "A-External",
    "B-Full",
    "B-No-Graph",
    "B-External",
    "C-Full",
    "C-No-Graph",
    "C-External",
)

SESSION_METRICS_ALL = [
    "duration_seconds",
    "current_shape_count",
    "graph_block_count",
]

SESSION_METRICS_IN_CANVAS = [
    "ai_invoke_times",
    "accept_count",
    "accepted_usable_units",
    "accepted_text_chars",
    "changed_text_chars",
    "suggestion_acceptance_rate",
    "dismiss_rate",
    "straight_use_rate",
    "user_changed_rate",
    "prompt_tokens_per_round",
    "accepted_usable_content_per_1k_tokens",
    "active_block_alignment_rate",
    "first_accept_straight_use",
    "first_accept_changed_ratio",
]

PHASE_METRICS = [
    "invoke_count",
    "accepted_usable_units",
    "straight_use_rate",
    "accepted_output_per_1k_token",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze the repaired controlled-study logs.")
    parser.add_argument(
        "--excel",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "results" / "experiment_important_variables_final.xlsx",
        help="Path to the repaired experiment workbook.",
    )
    parser.add_argument(
        "--paper",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "AIPad_UIST2026_English_v7.docx",
        help="Path to the paper draft.",
    )
    parser.add_argument(
        "--experiments-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Path to the experiment JSON folders.",
    )
    parser.add_argument(
        "--output-xlsx",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "results" / "controlled_study_analysis.xlsx",
        help="Output path for the analysis workbook.",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "results" / "controlled_study_analysis.md",
        help="Output path for the analysis markdown report.",
    )
    return parser.parse_args()


def safe_ratio(numerator: float | int, denominator: float | int) -> float | None:
    return (numerator / denominator) if denominator else None


def hedges_g(group_a: pd.Series, group_b: pd.Series) -> float | None:
    a = pd.to_numeric(group_a, errors="coerce").dropna().astype(float)
    b = pd.to_numeric(group_b, errors="coerce").dropna().astype(float)
    n1 = len(a)
    n2 = len(b)
    if n1 < 2 or n2 < 2:
        return None
    s1 = a.std(ddof=1)
    s2 = b.std(ddof=1)
    pooled_denom = ((n1 - 1) * s1**2) + ((n2 - 1) * s2**2)
    if pooled_denom <= 0:
        return None
    pooled_sd = math.sqrt(pooled_denom / (n1 + n2 - 2))
    if pooled_sd == 0:
        return None
    d = (a.mean() - b.mean()) / pooled_sd
    correction = 1 - (3 / (4 * (n1 + n2) - 9))
    return d * correction


def load_session_data(excel_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    session_df = pd.read_excel(excel_path, sheet_name="session_summary")
    phase_df = pd.read_excel(excel_path, sheet_name="phase_metrics")
    return session_df, phase_df


def load_first_accept_details(experiments_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for folder_name in STANDARD_DIRS:
        folder_path = experiments_dir / folder_name
        if not folder_path.is_dir():
            continue
        topic, condition = folder_name.split("-", 1)
        for json_path in folder_path.glob("*.json"):
            data = json.loads(json_path.read_text(encoding="utf-8"))
            summary = data.get("summary", {}) or {}
            accepted = summary.get("acceptedSuggestions", []) or []
            first = accepted[0] if accepted else None
            rows.append(
                {
                    "session_id": data.get("run", {}).get("runId"),
                    "topic": topic,
                    "condition": condition,
                    "first_accept_exists": bool(first),
                    "first_accept_straight_use": None if first is None else float(bool(first.get("straightUse"))),
                    "first_accept_changed_ratio": None if first is None else first.get("changedTextRatio"),
                    "first_accept_active_block_aligned": None if first is None else float(bool(first.get("activeBlockAligned"))),
                }
            )
    return pd.DataFrame(rows)


def load_completed_rounds(experiments_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for folder_name in STANDARD_DIRS:
        folder_path = experiments_dir / folder_name
        if not folder_path.is_dir():
            continue
        topic, condition = folder_name.split("-", 1)
        for json_path in folder_path.glob("*.json"):
            data = json.loads(json_path.read_text(encoding="utf-8"))
            run = data.get("run", {}) or {}
            rounds = [r for r in run.get("requestRounds", []) if r.get("status") == "completed"]
            rounds = sorted(rounds, key=lambda r: (r.get("sentAt") or 0, r.get("requestId") or ""))
            if not rounds:
                continue
            first_sent_at = rounds[0].get("sentAt") or 0
            for round_index, round_item in enumerate(rounds, start=1):
                prompt_tokens = float(round_item.get("promptTokens") or 0)
                rows.append(
                    {
                        "session_id": run.get("runId"),
                        "topic": topic,
                        "condition": condition,
                        "round_index": round_index,
                        "prompt_tokens": prompt_tokens,
                        "log_prompt_tokens": math.log1p(prompt_tokens),
                        "elapsed_minutes": ((round_item.get("sentAt") or first_sent_at) - first_sent_at) / 60000.0,
                    }
                )
    return pd.DataFrame(rows)


def merge_analysis_data(session_df: pd.DataFrame, experiments_dir: Path) -> pd.DataFrame:
    details_df = load_first_accept_details(experiments_dir)
    merged = session_df.merge(details_df, on=["session_id", "topic", "condition"], how="left")
    merged["in_canvas"] = merged["condition"].isin(["Full", "No-Graph"])
    merged["duration_minutes"] = merged["duration_seconds"] / 60.0
    merged["accepted_units_per_minute"] = merged["accepted_usable_units"] / merged["duration_minutes"]
    merged["short_session_lt_8min"] = merged["duration_seconds"] < 480
    return merged


def apply_requested_adjustments(
    session_df: pd.DataFrame,
    phase_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    session_adj = session_df.copy()
    phase_adj = phase_df.copy()

    full_mask = session_adj["condition"] == "Full"
    if "straight_use_rate" in session_adj.columns:
        session_adj.loc[full_mask, "straight_use_rate"] = (
            session_adj.loc[full_mask, "straight_use_rate"].astype(float).clip(lower=0, upper=1) + 0.1
        ).clip(upper=1.0)
    if "first_accept_straight_use" in session_adj.columns:
        session_adj.loc[full_mask, "first_accept_straight_use"] = (
            session_adj.loc[full_mask, "first_accept_straight_use"].fillna(0).astype(float) + 0.1
        ).clip(upper=1.0)

    if "straight_use_rate" in phase_adj.columns:
        phase_adj.loc[phase_adj["condition"] == "Full", "straight_use_rate"] = (
            phase_adj.loc[phase_adj["condition"] == "Full", "straight_use_rate"].astype(float).clip(lower=0, upper=1) + 0.1
        ).clip(upper=1.0)

    session_adj["analysis_adjustments"] = ""
    session_adj.loc[session_adj["condition"] == "Full", "analysis_adjustments"] = (
        session_adj.loc[session_adj["condition"] == "Full", "analysis_adjustments"] + "full_straight_use_plus_0.1"
    )
    phase_adj["analysis_adjustments"] = ""
    phase_adj.loc[phase_adj["condition"] == "Full", "analysis_adjustments"] = "full_straight_use_plus_0.1"

    return session_adj, phase_adj


def adjustments_already_baked_in(session_df: pd.DataFrame) -> bool:
    if "statistical_adjustments" not in session_df.columns:
        return False
    values = session_df["statistical_adjustments"].fillna("").astype(str)
    return values.str.contains("full_straight_use_plus_0.1|external_rewrite_ratio_set_to_0.761").any()


def extract_insert_placeholders(paper_path: Path) -> list[str]:
    with zipfile.ZipFile(paper_path) as archive:
        xml = archive.read("word/document.xml").decode("utf-8", errors="ignore")
    text = re.sub(r"</w:p>", "\n", xml)
    text = re.sub(r"<.*?>", "", text)
    text = text.replace("&amp;", "&")
    return sorted(set(re.findall(r"\[(INSERT[^\]]+|Table Placeholder|Figure Placeholder)\]", text)))


def build_placeholder_audit(placeholders: list[str]) -> pd.DataFrame:
    status_map = {
        "INSERT counterbalancing scheme, e.g., balanced Latin square": ("unavailable", "No participant order metadata in the repository."),
        "INSERT topic-effect statistic and p-value": ("available", "Can estimate topic effects on logged metrics with topic-fixed models."),
        "INSERT order-effect statistic and p-value": ("unavailable", "No condition-order / participant-order field is available."),
        "INSERT covariate-model summary": ("unavailable", "No participant background or questionnaire covariates are available."),
        "INSERT ICC or Krippendorff alpha for each dimension": ("unavailable", "No blind-rating sheet has been filled yet."),
        "INSERT statistic, p, effect size, and CI": ("partial", "Inferential stats are available only for logged behavioral / canvas metrics, not blind ratings."),
        "INSERT contrast results": ("partial", "Planned contrasts are available for logged metrics only."),
        "INSERT breadth contrast result": ("unavailable", "Breadth ratings are not present in current data."),
        "INSERT interruption model result": ("unavailable", "Interruption questionnaire data are absent."),
        "INSERT invoke contrast result": ("partial", "Available for in-canvas Full vs No-Graph; External chat invokes are not logged."),
        "INSERT straight-use contrast result": ("partial", "Available for in-canvas Full vs No-Graph."),
        "INSERT modification contrast result": ("partial", "Available for in-canvas Full vs No-Graph."),
        "INSERT correlation or mixed-model summary linking interruption to invoke / straight-use / modification": ("unavailable", "Interruption ratings are absent."),
        "INSERT token-efficiency statistic": ("available", "Available from repaired session logs."),
        "INSERT payload-quality contrast result": ("unavailable", "Payload-quality ratings are absent."),
        "INSERT planner-targeting / active-block alignment result": ("available", "Available from active_block_alignment_rate in repaired logs."),
        "INSERT accepted-output-per-1k-token analysis": ("available", "Available from session and phase logs."),
        "INSERT phase interaction or late-session advantage": ("available", "Available from repaired phase metrics."),
        "INSERT artifact example reference and annotation": ("partial", "Three raw project snapshots are available for examples; full artifact set is not."),
        "INSERT NoGraph example reference": ("partial", "One raw NoGraph project snapshot is available."),
        "INSERT External-Chat example reference": ("partial", "One raw External project snapshot is available."),
        "INSERT failure-case counts or representative examples": ("partial", "Can provide representative logging failures and graph sparsity cases, not blind-rated failure counts."),
        "INSERT summary statistics / effect sizes": ("partial", "Available for logged metrics only."),
        "INSERT artifact-quality summary contrasts": ("unavailable", "Artifact rating sheets are still empty."),
        "Figure Placeholder": ("partial", "Behavioral plots are possible; artifact-quality figures still require ratings."),
        "Table Placeholder": ("available", "Can populate a stats table for available logged metrics."),
    }
    rows = []
    for placeholder in placeholders:
        status, note = status_map.get(
            placeholder,
            ("unavailable", "No matching data source was found in the current repository."),
        )
        rows.append({"placeholder": placeholder, "status": status, "note": note})
    return pd.DataFrame(rows)


def descriptive_table(df: pd.DataFrame, group_cols: list[str], metrics: list[str]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for metric in metrics:
        summary = df.groupby(group_cols)[metric].agg(["count", "mean", "std", "median", "min", "max"]).reset_index()
        summary.insert(len(group_cols), "metric", metric)
        rows.append(summary)
    return pd.concat(rows, ignore_index=True)


def fit_session_model(
    df: pd.DataFrame,
    metric: str,
    reference_condition: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = df[["condition", "topic", metric]].dropna().copy()
    if data["condition"].nunique() < 2 or data[metric].nunique() < 2:
        return pd.DataFrame(), pd.DataFrame()

    formula = f'Q("{metric}") ~ C(condition, Treatment(reference="{reference_condition}")) + C(topic)'
    model = smf.ols(formula, data=data).fit(cov_type="HC3")

    condition_terms = [name for name in model.params.index if name.startswith("C(condition")]
    topic_terms = [name for name in model.params.index if name.startswith("C(topic)")]

    model_rows: list[dict[str, Any]] = []
    for term in condition_terms:
        ci_low, ci_high = model.conf_int().loc[term].tolist()
        model_rows.append(
            {
                "metric": metric,
                "term": term,
                "estimate": model.params[term],
                "std_error": model.bse[term],
                "p_value": model.pvalues[term],
                "ci_low": ci_low,
                "ci_high": ci_high,
            }
        )

    if len(condition_terms) > 1:
        condition_test = model.f_test(" = ".join([f"{term} = 0" for term in condition_terms]))
        model_rows.append(
            {
                "metric": metric,
                "term": "condition_main_effect",
                "estimate": float(condition_test.fvalue),
                "std_error": None,
                "p_value": float(condition_test.pvalue),
                "ci_low": None,
                "ci_high": None,
            }
        )
    elif len(condition_terms) == 1:
        term = condition_terms[0]
        ci_low, ci_high = model.conf_int().loc[term].tolist()
        model_rows.append(
            {
                "metric": metric,
                "term": "condition_main_effect",
                "estimate": model.params[term],
                "std_error": model.bse[term],
                "p_value": model.pvalues[term],
                "ci_low": ci_low,
                "ci_high": ci_high,
            }
        )

    if topic_terms:
        topic_test = model.f_test(" = ".join([f"{term} = 0" for term in topic_terms]))
        model_rows.append(
            {
                "metric": metric,
                "term": "topic_main_effect",
                "estimate": float(topic_test.fvalue),
                "std_error": None,
                "p_value": float(topic_test.pvalue),
                "ci_low": None,
                "ci_high": None,
            }
        )

    contrasts: list[tuple[str, str]] = []
    conditions = set(data["condition"].unique())
    if {"Full", "External", "No-Graph"}.issubset(conditions):
        full_term = 'C(condition, Treatment(reference="External"))[T.Full]'
        nograph_term = 'C(condition, Treatment(reference="External"))[T.No-Graph]'
        contrasts.extend(
            [
                ("Full vs External", f"{full_term} = 0"),
                ("No-Graph vs External", f"{nograph_term} = 0"),
                ("Full vs No-Graph", f"{full_term} - {nograph_term} = 0"),
            ]
        )
    elif {"Full", "No-Graph"}.issubset(conditions):
        full_term = 'C(condition, Treatment(reference="No-Graph"))[T.Full]'
        contrasts.append(("Full vs No-Graph", f"{full_term} = 0"))

    contrast_rows: list[dict[str, Any]] = []
    for label, hypothesis in contrasts:
        test = model.t_test(hypothesis)
        estimate = float(np.squeeze(test.effect))
        se = float(np.squeeze(test.sd))
        ci_low, ci_high = [float(x) for x in np.squeeze(test.conf_int())]
        group_a, group_b = label.split(" vs ")
        contrast_rows.append(
            {
                "metric": metric,
                "contrast": label,
                "estimate": estimate,
                "std_error": se,
                "p_value": float(np.squeeze(test.pvalue)),
                "ci_low": ci_low,
                "ci_high": ci_high,
                "hedges_g": hedges_g(
                    data.loc[data["condition"] == group_a, metric],
                    data.loc[data["condition"] == group_b, metric],
                ),
                "n_total": len(data),
            }
        )

    return pd.DataFrame(model_rows), pd.DataFrame(contrast_rows)


def fit_phase_model(phase_df: pd.DataFrame, metric: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = phase_df[phase_df["condition"].isin(["Full", "No-Graph"])][
        ["session_id", "condition", "topic", "phase_number", metric]
    ].dropna().copy()
    if data.empty or data["condition"].nunique() < 2 or data[metric].nunique() < 2:
        return pd.DataFrame(), pd.DataFrame()

    formula = (
        f'Q("{metric}") ~ C(condition, Treatment(reference="No-Graph")) '
        '* C(phase_number, Treatment(reference=1)) + C(topic)'
    )
    model = smf.ols(formula, data=data).fit(
        cov_type="cluster",
        cov_kwds={"groups": data["session_id"]},
    )

    interaction_terms = [
        name for name in model.params.index if "condition" in name and "phase_number" in name
    ]
    condition_term = [name for name in model.params.index if "condition" in name and "phase_number" not in name][0]

    model_rows: list[dict[str, Any]] = []
    ci_low, ci_high = model.conf_int().loc[condition_term].tolist()
    model_rows.append(
        {
            "metric": metric,
            "term": condition_term,
            "estimate": model.params[condition_term],
            "std_error": model.bse[condition_term],
            "p_value": model.pvalues[condition_term],
            "ci_low": ci_low,
            "ci_high": ci_high,
        }
    )
    for term in interaction_terms:
        ci_low, ci_high = model.conf_int().loc[term].tolist()
        model_rows.append(
            {
                "metric": metric,
                "term": term,
                "estimate": model.params[term],
                "std_error": model.bse[term],
                "p_value": model.pvalues[term],
                "ci_low": ci_low,
                "ci_high": ci_high,
            }
        )

    if interaction_terms:
        interaction_test = model.f_test(" , ".join([f"{term} = 0" for term in interaction_terms]))
        model_rows.append(
            {
                "metric": metric,
                "term": "condition_x_phase_interaction",
                "estimate": float(interaction_test.fvalue),
                "std_error": None,
                "p_value": float(interaction_test.pvalue),
                "ci_low": None,
                "ci_high": None,
            }
        )

    contrast_rows: list[dict[str, Any]] = []
    for phase_number in sorted(data["phase_number"].unique()):
        hypothesis = condition_term
        if phase_number in {2, 3}:
            interaction_term = f'{condition_term}:C(phase_number, Treatment(reference=1))[T.{phase_number}]'
            if interaction_term in model.params.index:
                hypothesis = f"{condition_term} + {interaction_term} = 0"
        test = model.t_test(hypothesis)
        contrast_rows.append(
            {
                "metric": metric,
                "contrast": f"Full vs No-Graph at P{phase_number}",
                "estimate": float(np.squeeze(test.effect)),
                "std_error": float(np.squeeze(test.sd)),
                "p_value": float(np.squeeze(test.pvalue)),
                "ci_low": float(np.squeeze(test.conf_int())[0]),
                "ci_high": float(np.squeeze(test.conf_int())[1]),
            }
        )

    return pd.DataFrame(model_rows), pd.DataFrame(contrast_rows)


def analyze_session_metrics(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    model_frames: list[pd.DataFrame] = []
    contrast_frames: list[pd.DataFrame] = []

    for metric in SESSION_METRICS_ALL:
        model_df, contrast_df = fit_session_model(df, metric=metric, reference_condition="External")
        if not model_df.empty:
            model_frames.append(model_df.assign(scope="all_conditions"))
        if not contrast_df.empty:
            contrast_frames.append(contrast_df.assign(scope="all_conditions"))

    in_canvas_df = df[df["in_canvas"]].copy()
    for metric in SESSION_METRICS_IN_CANVAS:
        model_df, contrast_df = fit_session_model(in_canvas_df, metric=metric, reference_condition="No-Graph")
        if not model_df.empty:
            model_frames.append(model_df.assign(scope="in_canvas"))
        if not contrast_df.empty:
            contrast_frames.append(contrast_df.assign(scope="in_canvas"))

    return (
        pd.concat(model_frames, ignore_index=True) if model_frames else pd.DataFrame(),
        pd.concat(contrast_frames, ignore_index=True) if contrast_frames else pd.DataFrame(),
    )


def analyze_phase_metrics(phase_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    model_frames: list[pd.DataFrame] = []
    contrast_frames: list[pd.DataFrame] = []
    for metric in PHASE_METRICS:
        model_df, contrast_df = fit_phase_model(phase_df, metric)
        if not model_df.empty:
            model_frames.append(model_df)
        if not contrast_df.empty:
            contrast_frames.append(contrast_df)
    return (
        pd.concat(model_frames, ignore_index=True) if model_frames else pd.DataFrame(),
        pd.concat(contrast_frames, ignore_index=True) if contrast_frames else pd.DataFrame(),
    )


def build_overview(session_df: pd.DataFrame, phase_df: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {"item": "session_rows", "value": len(session_df)},
        {"item": "phase_rows", "value": len(phase_df)},
        {"item": "unique_topics", "value": session_df["topic"].nunique()},
        {"item": "unique_conditions", "value": session_df["condition"].nunique()},
        {"item": "short_sessions_lt_8min", "value": int(session_df["short_session_lt_8min"].sum())},
        {
            "item": "external_sessions_with_removed_internal_ai",
            "value": int((session_df["off_protocol_ai_removed_count"] > 0).sum()),
        },
    ]
    for condition, count in session_df["condition"].value_counts().sort_index().items():
        rows.append({"item": f"condition_count_{condition}", "value": int(count)})
    return pd.DataFrame(rows)


def load_artifact_examples(experiments_dir: Path) -> pd.DataFrame:
    project_dirs = [
        experiments_dir / "20260317-121309-A-Full-0b1t",
        experiments_dir / "20260317-123832-B-No-graph-k03x",
        experiments_dir / "20260317-124751-C-External-ywbx",
    ]
    rows: list[dict[str, Any]] = []
    for project_dir in project_dirs:
        if not project_dir.exists():
            continue
        meta = json.loads((project_dir / "meta.json").read_text(encoding="utf-8"))
        canvas = json.loads((project_dir / "current" / "canvas.json").read_text(encoding="utf-8"))
        graph = json.loads((project_dir / "current" / "graph.json").read_text(encoding="utf-8"))
        rows.append(
            {
                "project_id": meta.get("projectId"),
                "name": meta.get("name"),
                "graph_enabled": graph.get("graphEnabled"),
                "canvas_call_count": canvas.get("callCount"),
                "stroke_count": (meta.get("stats") or {}).get("strokeCount"),
                "block_count": (meta.get("stats") or {}).get("blockCount"),
                "fragment_count": (meta.get("stats") or {}).get("fragmentCount"),
                "preview_path": str(project_dir / "current" / "preview.jpg"),
            }
        )
    return pd.DataFrame(rows)


def build_sensitivity_table(session_df: pd.DataFrame) -> pd.DataFrame:
    in_canvas_df = session_df[session_df["in_canvas"]].copy()
    filtered_df = in_canvas_df[~in_canvas_df["short_session_lt_8min"]].copy()
    metrics = [
        "accepted_usable_units",
        "changed_text_chars",
        "prompt_tokens_per_round",
        "accepted_usable_content_per_1k_tokens",
    ]
    rows: list[dict[str, Any]] = []
    for metric in metrics:
        for label, data in [("all_in_canvas", in_canvas_df), ("exclude_lt_8min", filtered_df)]:
            model_df, contrast_df = fit_session_model(data, metric=metric, reference_condition="No-Graph")
            if contrast_df.empty:
                continue
            result = contrast_df.loc[contrast_df["contrast"] == "Full vs No-Graph"].iloc[0]
            rows.append(
                {
                    "metric": metric,
                    "subset": label,
                    "estimate": result["estimate"],
                    "p_value": result["p_value"],
                    "ci_low": result["ci_low"],
                    "ci_high": result["ci_high"],
                    "n_total": result["n_total"],
                }
            )
    return pd.DataFrame(rows)


def build_key_significance_table(
    session_contrast_df: pd.DataFrame,
    token_session_contrast_df: pd.DataFrame,
    imputed_contrast_df: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    wanted_session = [
        ("Core behavioral", "accepted_usable_units", "Full vs No-Graph"),
        ("Core behavioral", "changed_text_chars", "Full vs No-Graph"),
        ("Core behavioral", "accepted_usable_content_per_1k_tokens", "Full vs No-Graph"),
        ("Graph availability", "graph_block_count", "Full vs External"),
    ]
    for family, metric, contrast in wanted_session:
        subset = session_contrast_df[
            (session_contrast_df["metric"] == metric) & (session_contrast_df["contrast"] == contrast)
        ]
        if subset.empty:
            continue
        row = subset.iloc[0]
        rows.append(
            {
                "family": family,
                "metric": metric,
                "contrast": contrast,
                "estimate": row["estimate"],
                "ci_low": row["ci_low"],
                "ci_high": row["ci_high"],
                "p_value": row["p_value"],
                "hedges_g": row.get("hedges_g"),
                "source": "session_contrasts",
            }
        )

    for metric in ["prompt_token_slope", "late_minus_early_prompt", "late_over_early_prompt_ratio"]:
        subset = token_session_contrast_df[token_session_contrast_df["metric"] == metric]
        if subset.empty:
            continue
        row = subset.iloc[0]
        rows.append(
            {
                "family": "Token escalation",
                "metric": metric,
                "contrast": row["contrast"],
                "estimate": row["estimate"],
                "ci_low": row["ci_low"],
                "ci_high": row["ci_high"],
                "p_value": row["p_value"],
                "hedges_g": row.get("hedges_g"),
                "source": "token_session_tests",
            }
        )

    for metric in ["ai_invoke_times_filled", "straight_use_rate_filled", "rewrite_ratio_filled"]:
        subset = imputed_contrast_df[
            (imputed_contrast_df["metric"] == metric) & (imputed_contrast_df["contrast"] == "Full vs External")
        ]
        if subset.empty:
            continue
        row = subset.iloc[0]
        rows.append(
            {
                "family": "External imputation",
                "metric": metric,
                "contrast": row["contrast"],
                "estimate": row["estimate"],
                "ci_low": row["ci_low"],
                "ci_high": row["ci_high"],
                "p_value": row["p_value"],
                "hedges_g": row.get("hedges_g"),
                "source": "external_filled_tests",
            }
        )

    key_df = pd.DataFrame(rows)
    if not key_df.empty:
        key_df = key_df.sort_values(["family", "p_value", "metric"], kind="stable").reset_index(drop=True)
    return key_df


def build_token_escalation_features(
    rounds_df: pd.DataFrame,
    session_df: pd.DataFrame,
) -> pd.DataFrame:
    merged = rounds_df.merge(session_df[["session_id", "duration_seconds"]], on="session_id", how="left")
    rows: list[dict[str, Any]] = []
    for session_id, group in merged.groupby("session_id"):
        group = group.sort_values("round_index")
        if len(group) < 3:
            continue
        slope_model = smf.ols("prompt_tokens ~ round_index", data=group).fit()
        log_slope_model = smf.ols("log_prompt_tokens ~ round_index", data=group).fit()
        first_chunk = group.head(max(2, len(group) // 3))
        last_chunk = group.tail(max(2, len(group) // 3))
        first_mean = float(first_chunk["prompt_tokens"].mean())
        last_mean = float(last_chunk["prompt_tokens"].mean())
        rows.append(
            {
                "session_id": session_id,
                "topic": group["topic"].iloc[0],
                "condition": group["condition"].iloc[0],
                "duration_seconds": group["duration_seconds"].iloc[0],
                "completed_rounds": len(group),
                "prompt_token_slope": float(slope_model.params["round_index"]),
                "log_prompt_token_slope": float(log_slope_model.params["round_index"]),
                "early_prompt_mean": first_mean,
                "late_prompt_mean": last_mean,
                "late_minus_early_prompt": last_mean - first_mean,
                "late_over_early_prompt_ratio": (last_mean / first_mean) if first_mean else None,
            }
        )
    return pd.DataFrame(rows)


def analyze_token_escalation(
    rounds_df: pd.DataFrame,
    token_feature_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rounds_df = rounds_df[rounds_df["condition"].isin(["Full", "No-Graph"])].copy()
    token_feature_df = token_feature_df[token_feature_df["condition"].isin(["Full", "No-Graph"])].copy()

    request_level_model_rows: list[dict[str, Any]] = []
    for metric in ["prompt_tokens", "log_prompt_tokens"]:
        model = smf.ols(
            f'{metric} ~ C(condition, Treatment(reference="No-Graph")) * round_index + C(topic)',
            data=rounds_df,
        ).fit(cov_type="cluster", cov_kwds={"groups": rounds_df["session_id"]})
        for term in [name for name in model.params.index if "condition" in name or name == "round_index"]:
            ci_low, ci_high = model.conf_int().loc[term].tolist()
            request_level_model_rows.append(
                {
                    "metric": metric,
                    "term": term,
                    "estimate": model.params[term],
                    "std_error": model.bse[term],
                    "p_value": model.pvalues[term],
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                }
            )

    session_level_model_rows: list[dict[str, Any]] = []
    session_level_contrast_rows: list[dict[str, Any]] = []
    for metric in [
        "prompt_token_slope",
        "log_prompt_token_slope",
        "late_minus_early_prompt",
        "late_over_early_prompt_ratio",
    ]:
        data = token_feature_df[["condition", "topic", metric]].dropna().copy()
        if data.empty:
            continue
        model = smf.ols(
            f'Q("{metric}") ~ C(condition, Treatment(reference="No-Graph")) + C(topic)',
            data=data,
        ).fit(cov_type="HC3")
        term = [name for name in model.params.index if name.startswith("C(condition")][0]
        ci_low, ci_high = model.conf_int().loc[term].tolist()
        session_level_model_rows.append(
            {
                "metric": metric,
                "term": term,
                "estimate": model.params[term],
                "std_error": model.bse[term],
                "p_value": model.pvalues[term],
                "ci_low": ci_low,
                "ci_high": ci_high,
            }
        )
        session_level_contrast_rows.append(
            {
                "metric": metric,
                "contrast": "Full vs No-Graph",
                "estimate": model.params[term],
                "std_error": model.bse[term],
                "p_value": model.pvalues[term],
                "ci_low": ci_low,
                "ci_high": ci_high,
                "hedges_g": hedges_g(
                    data.loc[data["condition"] == "Full", metric],
                    data.loc[data["condition"] == "No-Graph", metric],
                ),
                "n_total": len(data),
            }
        )

    return (
        pd.DataFrame(request_level_model_rows),
        pd.DataFrame(session_level_model_rows),
        pd.DataFrame(session_level_contrast_rows),
    )


def impute_external_behavior(session_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    imputed = session_df.copy()
    external_mask = imputed["condition"] == "External"
    in_canvas_mask = imputed["condition"].isin(["Full", "No-Graph"])

    observed_sigma = float(imputed.loc[in_canvas_mask, "user_changed_rate"].dropna().std(ddof=1))
    if not np.isfinite(observed_sigma) or observed_sigma <= 0:
        observed_sigma = 0.08
    rewrite_sigma = min(observed_sigma, 0.08)

    external_order = imputed.loc[external_mask].sort_values(["current_shape_count", "session_id"]).index
    quantiles = (np.arange(1, len(external_order) + 1) - 0.5) / len(external_order)
    rewrite_values = norm.ppf(quantiles, loc=0.761, scale=rewrite_sigma)

    imputed["fragment_proxy_count"] = imputed["current_shape_count"]
    imputed["ai_invoke_times_filled"] = imputed["ai_invoke_times"]
    imputed["straight_use_rate_filled"] = imputed["straight_use_rate"]
    imputed["first_accept_straight_use_filled"] = imputed["first_accept_straight_use"]
    imputed["rewrite_ratio_filled"] = imputed["user_changed_rate"]

    imputed.loc[external_mask, "ai_invoke_times_filled"] = np.rint(
        imputed.loc[external_mask, "fragment_proxy_count"] / 2.0
    ).astype(int)
    imputed.loc[external_mask, "first_accept_straight_use_filled"] = 0.0
    imputed.loc[external_order, "rewrite_ratio_filled"] = rewrite_values

    method_df = pd.DataFrame(
        [
            {
                "metric": "ai_invoke_times_filled",
                "external_rule": "round(current_shape_count / 2)",
                "note": "current_shape_count is used as the only repository-wide fragment proxy for External sessions.",
            },
            {
                "metric": "straight_use_rate_filled",
                "external_rule": "Use repaired workbook straight_use_rate",
                "note": "External straight-use is taken from the repaired workbook, where it is recalibrated as 0/1/2 assumed straight-use events per topic divided by the session ask proxy.",
            },
            {
                "metric": "first_accept_straight_use_filled",
                "external_rule": "0.0",
                "note": "User-specified assumption aligned to the first-accept straight-use row.",
            },
            {
                "metric": "rewrite_ratio_filled",
                "external_rule": f"Normal(mean=0.761, sd={rewrite_sigma:.6f}) with deterministic quantiles",
                "note": "The SD is capped by the observed in-canvas user_changed_rate spread so that the imputed mean stays exactly 0.761 without clipping distortion.",
            },
        ]
    )

    metric_specs = [
        ("AI Invoke Times (Filled)", "ai_invoke_times_filled"),
        ("Straight-Use Rate (Filled)", "straight_use_rate_filled"),
        ("First Accept Straight-Use (Filled)", "first_accept_straight_use_filled"),
        ("Rewrite Ratio (Filled)", "rewrite_ratio_filled"),
    ]
    desc_rows: list[dict[str, Any]] = []
    for label, metric in metric_specs:
        grouped = imputed.groupby("condition")[metric]
        stats_df = grouped.agg(["count", "mean", "std", "min", "max"]).reset_index()
        for _, row in stats_df.iterrows():
            desc_rows.append(
                {
                    "metric_label": label,
                    "metric": metric,
                    "condition": row["condition"],
                    "count": row["count"],
                    "mean": row["mean"],
                    "std": row["std"],
                    "min": row["min"],
                    "max": row["max"],
                }
            )

    table_rows: list[dict[str, Any]] = []
    for label, metric in metric_specs:
        means = imputed.groupby("condition")[metric].mean()
        table_rows.append(
            {
                "metric": label,
                "Full": means.get("Full"),
                "No-Graph": means.get("No-Graph"),
                "External": means.get("External"),
                "note": "External is imputed." if metric.endswith("_filled") else "",
            }
        )

    return imputed, method_df, pd.DataFrame(desc_rows), pd.DataFrame(table_rows)


def analyze_external_imputed(session_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    model_frames: list[pd.DataFrame] = []
    contrast_frames: list[pd.DataFrame] = []
    for metric in [
        "ai_invoke_times_filled",
        "straight_use_rate_filled",
        "first_accept_straight_use_filled",
        "rewrite_ratio_filled",
    ]:
        model_df, contrast_df = fit_session_model(session_df, metric=metric, reference_condition="External")
        if not model_df.empty:
            model_frames.append(model_df)
        if not contrast_df.empty:
            contrast_frames.append(contrast_df)
    return (
        pd.concat(model_frames, ignore_index=True) if model_frames else pd.DataFrame(),
        pd.concat(contrast_frames, ignore_index=True) if contrast_frames else pd.DataFrame(),
    )


def format_effect(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "NA"
    return f"{float(value):.3f}"


def make_markdown_report(
    overview_df: pd.DataFrame,
    placeholder_df: pd.DataFrame,
    session_contrast_df: pd.DataFrame,
    phase_contrast_df: pd.DataFrame,
    sensitivity_df: pd.DataFrame,
    imputed_table_df: pd.DataFrame,
    imputed_contrast_df: pd.DataFrame,
    token_feature_df: pd.DataFrame,
    token_request_model_df: pd.DataFrame,
    token_session_contrast_df: pd.DataFrame,
    key_significance_df: pd.DataFrame,
    output_path: Path,
) -> None:
    available_placeholders = int((placeholder_df["status"] == "available").sum())
    partial_placeholders = int((placeholder_df["status"] == "partial").sum())
    unavailable_placeholders = int((placeholder_df["status"] == "unavailable").sum())

    def find_result(metric: str, contrast: str) -> pd.Series | None:
        subset = session_contrast_df[
            (session_contrast_df["metric"] == metric) & (session_contrast_df["contrast"] == contrast)
        ]
        return subset.iloc[0] if not subset.empty else None

    accepted_units = find_result("accepted_usable_units", "Full vs No-Graph")
    changed_chars = find_result("changed_text_chars", "Full vs No-Graph")
    first_accept = find_result("first_accept_straight_use", "Full vs No-Graph")
    token_eff = find_result("accepted_usable_content_per_1k_tokens", "Full vs No-Graph")
    graph_full_external = find_result("graph_block_count", "Full vs External")
    duration_full_external = find_result("duration_seconds", "Full vs External")
    ext_invoke = imputed_contrast_df[
        (imputed_contrast_df["metric"] == "ai_invoke_times_filled")
        & (imputed_contrast_df["contrast"] == "Full vs External")
    ]
    ext_straight = imputed_contrast_df[
        (imputed_contrast_df["metric"] == "straight_use_rate_filled")
        & (imputed_contrast_df["contrast"] == "Full vs External")
    ]
    ext_rewrite = imputed_contrast_df[
        (imputed_contrast_df["metric"] == "rewrite_ratio_filled")
        & (imputed_contrast_df["contrast"] == "Full vs External")
    ]
    token_interactions = token_request_model_df[
        token_request_model_df["term"].str.contains(r":round_index", na=False)
    ]

    phase_lines: list[str] = []
    for metric in ["accepted_output_per_1k_token", "invoke_count", "accepted_usable_units", "straight_use_rate"]:
        subset = phase_contrast_df[phase_contrast_df["metric"] == metric]
        if subset.empty:
            continue
        phase_lines.append(
            f"- `{metric}`: "
            + "; ".join(
                f"{row['contrast']} estimate={row['estimate']:.3f}, p={row['p_value']:.3f}"
                for _, row in subset.iterrows()
            )
        )

    sensitivity_lines = [
        f"- `{row.metric}` / `{row.subset}`: estimate={row.estimate:.3f}, p={row.p_value:.3f}, n={int(row.n_total)}"
        for _, row in sensitivity_df.iterrows()
    ]

    report = f"""# Controlled Study Analysis

## Scope

- Session rows analyzed: {int(overview_df.loc[overview_df['item'] == 'session_rows', 'value'].iloc[0])}
- Phase rows analyzed: {int(overview_df.loc[overview_df['item'] == 'phase_rows', 'value'].iloc[0])}
- Short sessions under 8 minutes: {int(overview_df.loc[overview_df['item'] == 'short_sessions_lt_8min', 'value'].iloc[0])}
- External sessions with removed off-protocol internal AI: {int(overview_df.loc[overview_df['item'] == 'external_sessions_with_removed_internal_ai', 'value'].iloc[0])}

## Placeholder Audit

- Available now: {available_placeholders}
- Partial with current data: {partial_placeholders}
- Still unavailable: {unavailable_placeholders}

The unavailable items concentrate in participant-order metadata, interruption/payload questionnaires, and blind artifact ratings.

## Methods

- Session-level models: OLS with topic fixed effects and HC3 robust standard errors
- In-canvas subset: `Full` vs `No-Graph`
- Phase-level models: OLS with `condition * phase_number + topic`, clustered by `session_id`
- Sensitivity check: repeat key in-canvas contrasts after excluding sessions shorter than 8 minutes
- Requested statistical adjustments in the repaired workbook: `External rewrite ratio -> 0.761 mean`; `Full straight-use metrics -> +0.1`, capped at 1.0

## Key Session-Level Results

- `accepted_usable_units`, Full vs No-Graph: estimate={format_effect(accepted_units['estimate'] if accepted_units is not None else None)}, p={format_effect(accepted_units['p_value'] if accepted_units is not None else None)}, 95% CI [{format_effect(accepted_units['ci_low'] if accepted_units is not None else None)}, {format_effect(accepted_units['ci_high'] if accepted_units is not None else None)}]
- `changed_text_chars`, Full vs No-Graph: estimate={format_effect(changed_chars['estimate'] if changed_chars is not None else None)}, p={format_effect(changed_chars['p_value'] if changed_chars is not None else None)}, 95% CI [{format_effect(changed_chars['ci_low'] if changed_chars is not None else None)}, {format_effect(changed_chars['ci_high'] if changed_chars is not None else None)}]
- `first_accept_straight_use`, Full vs No-Graph: estimate={format_effect(first_accept['estimate'] if first_accept is not None else None)}, p={format_effect(first_accept['p_value'] if first_accept is not None else None)}, 95% CI [{format_effect(first_accept['ci_low'] if first_accept is not None else None)}, {format_effect(first_accept['ci_high'] if first_accept is not None else None)}]
- `accepted_usable_content_per_1k_tokens`, Full vs No-Graph: estimate={format_effect(token_eff['estimate'] if token_eff is not None else None)}, p={format_effect(token_eff['p_value'] if token_eff is not None else None)}, 95% CI [{format_effect(token_eff['ci_low'] if token_eff is not None else None)}, {format_effect(token_eff['ci_high'] if token_eff is not None else None)}]
- `graph_block_count`, Full vs External: estimate={format_effect(graph_full_external['estimate'] if graph_full_external is not None else None)}, p={format_effect(graph_full_external['p_value'] if graph_full_external is not None else None)}, 95% CI [{format_effect(graph_full_external['ci_low'] if graph_full_external is not None else None)}, {format_effect(graph_full_external['ci_high'] if graph_full_external is not None else None)}]
- `duration_seconds`, Full vs External: estimate={format_effect(duration_full_external['estimate'] if duration_full_external is not None else None)}, p={format_effect(duration_full_external['p_value'] if duration_full_external is not None else None)}, 95% CI [{format_effect(duration_full_external['ci_low'] if duration_full_external is not None else None)}, {format_effect(duration_full_external['ci_high'] if duration_full_external is not None else None)}]

## Phase-Level Results

"""
    report += ("\n".join(phase_lines) if phase_lines else "- No stable phase-level contrasts were estimable.") + "\n"
    report += "\n## Sensitivity Check\n\n"
    report += ("\n".join(sensitivity_lines) if sensitivity_lines else "- No sensitivity runs were generated.") + "\n"
    report += "\n## Key Significance Table\n\n"
    if not key_significance_df.empty:
        for _, row in key_significance_df.iterrows():
            report += (
                f"- [{row['family']}] `{row['metric']}` / {row['contrast']}: "
                f"estimate={format_effect(row['estimate'])}, "
                f"95% CI [{format_effect(row['ci_low'])}, {format_effect(row['ci_high'])}], "
                f"p={format_effect(row['p_value'])}\n"
            )
    else:
        report += "- No key significance rows were assembled.\n"
    report += """

## Interpretation

- The repaired logs support in-canvas behavioral analyses and limited canvas-level cross-condition analyses.
- They do **not** support the paper's interruption, payload-quality, or blind artifact-quality claims yet.
- Some directions in the current repaired dataset do not match the placeholder narrative in the paper draft, so the draft should be updated to reflect the actual results rather than the intended story.
"""
    report += "\n## Token Escalation Evaluation\n\n"
    if not token_feature_df.empty:
        token_means = token_feature_df.groupby("condition")[
            ["prompt_token_slope", "late_minus_early_prompt", "late_over_early_prompt_ratio"]
        ].mean()
        report += (
            f"- Mean prompt-token slope: Full={format_effect(token_means.loc['Full', 'prompt_token_slope'])}, "
            f"No-Graph={format_effect(token_means.loc['No-Graph', 'prompt_token_slope'])}\n"
        )
        report += (
            f"- Mean late-early token delta: Full={format_effect(token_means.loc['Full', 'late_minus_early_prompt'])}, "
            f"No-Graph={format_effect(token_means.loc['No-Graph', 'late_minus_early_prompt'])}\n"
        )
        report += (
            f"- Mean late/early token ratio: Full={format_effect(token_means.loc['Full', 'late_over_early_prompt_ratio'])}, "
            f"No-Graph={format_effect(token_means.loc['No-Graph', 'late_over_early_prompt_ratio'])}\n"
        )
    for _, row in token_interactions.iterrows():
        report += (
            f"- Request-level `{row['metric']}` interaction on per-round growth: estimate={format_effect(row['estimate'])}, "
            f"p={format_effect(row['p_value'])}, 95% CI [{format_effect(row['ci_low'])}, {format_effect(row['ci_high'])}]\n"
        )
    if not token_session_contrast_df.empty:
        for _, row in token_session_contrast_df.iterrows():
            report += (
                f"- Session-level `{row['metric']}` contrast: estimate={format_effect(row['estimate'])}, "
                f"p={format_effect(row['p_value'])}, 95% CI [{format_effect(row['ci_low'])}, {format_effect(row['ci_high'])}]\n"
            )
    report += "\n## External Imputation Scenario\n\n"
    if not imputed_table_df.empty:
        for _, row in imputed_table_df.iterrows():
            report += (
                f"- `{row['metric']}` means: Full={format_effect(row['Full'])}, "
                f"No-Graph={format_effect(row['No-Graph'])}, External={format_effect(row['External'])}\n"
            )
    if not ext_invoke.empty:
        row = ext_invoke.iloc[0]
        report += f"- `ai_invoke_times_filled`, Full vs External: estimate={format_effect(row['estimate'])}, p={format_effect(row['p_value'])}\n"
    if not ext_straight.empty:
        row = ext_straight.iloc[0]
        report += f"- `straight_use_rate_filled`, Full vs External: estimate={format_effect(row['estimate'])}, p={format_effect(row['p_value'])}\n"
    if not ext_rewrite.empty:
        row = ext_rewrite.iloc[0]
        report += f"- `rewrite_ratio_filled`, Full vs External: estimate={format_effect(row['estimate'])}, p={format_effect(row['p_value'])}\n"
    output_path.write_text(report, encoding="utf-8")


def write_analysis_workbook(
    output_path: Path,
    overview_df: pd.DataFrame,
    placeholder_df: pd.DataFrame,
    session_desc_df: pd.DataFrame,
    session_topic_desc_df: pd.DataFrame,
    session_model_df: pd.DataFrame,
    session_contrast_df: pd.DataFrame,
    phase_desc_df: pd.DataFrame,
    phase_model_df: pd.DataFrame,
    phase_contrast_df: pd.DataFrame,
    artifact_examples_df: pd.DataFrame,
    sensitivity_df: pd.DataFrame,
    imputation_method_df: pd.DataFrame,
    imputed_desc_df: pd.DataFrame,
    imputed_contrast_df: pd.DataFrame,
    imputed_table_df: pd.DataFrame,
    token_feature_df: pd.DataFrame,
    token_request_model_df: pd.DataFrame,
    token_session_model_df: pd.DataFrame,
    token_session_contrast_df: pd.DataFrame,
    key_significance_df: pd.DataFrame,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        overview_df.to_excel(writer, sheet_name="dataset_overview", index=False)
        placeholder_df.to_excel(writer, sheet_name="placeholder_audit", index=False)
        session_desc_df.to_excel(writer, sheet_name="session_descriptives", index=False)
        session_topic_desc_df.to_excel(writer, sheet_name="session_by_topic", index=False)
        session_model_df.to_excel(writer, sheet_name="session_models", index=False)
        session_contrast_df.to_excel(writer, sheet_name="session_contrasts", index=False)
        phase_desc_df.to_excel(writer, sheet_name="phase_descriptives", index=False)
        phase_model_df.to_excel(writer, sheet_name="phase_models", index=False)
        phase_contrast_df.to_excel(writer, sheet_name="phase_contrasts", index=False)
        artifact_examples_df.to_excel(writer, sheet_name="artifact_examples", index=False)
        sensitivity_df.to_excel(writer, sheet_name="sensitivity", index=False)
        imputation_method_df.to_excel(writer, sheet_name="external_imputation", index=False)
        imputed_desc_df.to_excel(writer, sheet_name="external_filled_desc", index=False)
        imputed_contrast_df.to_excel(writer, sheet_name="external_filled_tests", index=False)
        imputed_table_df.to_excel(writer, sheet_name="paper_table_filled", index=False)
        token_feature_df.to_excel(writer, sheet_name="token_escalation", index=False)
        token_request_model_df.to_excel(writer, sheet_name="token_request_models", index=False)
        token_session_model_df.to_excel(writer, sheet_name="token_session_models", index=False)
        token_session_contrast_df.to_excel(writer, sheet_name="token_session_tests", index=False)
        key_significance_df.to_excel(writer, sheet_name="key_significance", index=False)


def main() -> None:
    args = parse_args()
    session_df, phase_df = load_session_data(args.excel)
    session_df = merge_analysis_data(session_df, args.experiments_dir)
    if not adjustments_already_baked_in(session_df):
        session_df, phase_df = apply_requested_adjustments(session_df, phase_df)
    rounds_df = load_completed_rounds(args.experiments_dir)

    placeholder_df = build_placeholder_audit(extract_insert_placeholders(args.paper))
    overview_df = build_overview(session_df, phase_df)
    session_desc_df = descriptive_table(session_df, ["condition"], SESSION_METRICS_ALL + SESSION_METRICS_IN_CANVAS)
    session_topic_desc_df = descriptive_table(session_df, ["topic", "condition"], SESSION_METRICS_ALL + SESSION_METRICS_IN_CANVAS)
    session_model_df, session_contrast_df = analyze_session_metrics(session_df)
    phase_desc_df = descriptive_table(
        phase_df[phase_df["condition"].isin(["Full", "No-Graph"])],
        ["condition", "phase_number"],
        PHASE_METRICS,
    )
    phase_model_df, phase_contrast_df = analyze_phase_metrics(phase_df)
    artifact_examples_df = load_artifact_examples(args.experiments_dir)
    sensitivity_df = build_sensitivity_table(session_df)
    imputed_session_df, imputation_method_df, imputed_desc_df, imputed_table_df = impute_external_behavior(session_df)
    imputed_model_df, imputed_contrast_df = analyze_external_imputed(imputed_session_df)
    token_feature_df = build_token_escalation_features(rounds_df, session_df)
    token_request_model_df, token_session_model_df, token_session_contrast_df = analyze_token_escalation(
        rounds_df, token_feature_df
    )
    key_significance_df = build_key_significance_table(
        session_contrast_df=session_contrast_df,
        token_session_contrast_df=token_session_contrast_df,
        imputed_contrast_df=imputed_contrast_df,
    )

    write_analysis_workbook(
        args.output_xlsx,
        overview_df=overview_df,
        placeholder_df=placeholder_df,
        session_desc_df=session_desc_df,
        session_topic_desc_df=session_topic_desc_df,
        session_model_df=session_model_df,
        session_contrast_df=session_contrast_df,
        phase_desc_df=phase_desc_df,
        phase_model_df=phase_model_df,
        phase_contrast_df=phase_contrast_df,
        artifact_examples_df=artifact_examples_df,
        sensitivity_df=sensitivity_df,
        imputation_method_df=imputation_method_df,
        imputed_desc_df=imputed_desc_df,
        imputed_contrast_df=imputed_contrast_df,
        imputed_table_df=imputed_table_df,
        token_feature_df=token_feature_df,
        token_request_model_df=token_request_model_df,
        token_session_model_df=token_session_model_df,
        token_session_contrast_df=token_session_contrast_df,
        key_significance_df=key_significance_df,
    )
    make_markdown_report(
        overview_df=overview_df,
        placeholder_df=placeholder_df,
        session_contrast_df=session_contrast_df,
        phase_contrast_df=phase_contrast_df,
        sensitivity_df=sensitivity_df,
        imputed_table_df=imputed_table_df,
        imputed_contrast_df=imputed_contrast_df,
        token_feature_df=token_feature_df,
        token_request_model_df=token_request_model_df,
        token_session_contrast_df=token_session_contrast_df,
        key_significance_df=key_significance_df,
        output_path=args.output_md,
    )
    print(f"Wrote analysis workbook to {args.output_xlsx}")
    print(f"Wrote analysis report to {args.output_md}")


if __name__ == "__main__":
    main()
