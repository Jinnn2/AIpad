from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Rectangle
from PIL import Image
from statsmodels.nonparametric.smoothers_lowess import lowess

from analyze_controlled_study import load_completed_rounds


PALETTE = {
    "Full": "#0f766e",
    "No-Graph": "#d97706",
    "External": "#475569",
}

PAPER_BG = "#f7f3eb"
PANEL_BG = "#ffffff"
TEXT_MUTED = "#6b7280"
TEXT_DARK = "#111827"
GRID = "#e5e7eb"
BORDER = "#d1d5db"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate paper-ready figures from the repaired study data.")
    parser.add_argument(
        "--analysis-xlsx",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "results" / "controlled_study_analysis.xlsx",
        help="Path to the analysis workbook.",
    )
    parser.add_argument(
        "--experiment-xlsx",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "results" / "experiment_important_variables_final.xlsx",
        help="Path to the repaired experiment workbook.",
    )
    parser.add_argument(
        "--experiments-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Path to the experiments directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "results" / "figures",
        help="Directory for exported figures.",
    )
    return parser.parse_args()


def set_plot_style() -> None:
    sns.set_theme(
        style="whitegrid",
        context="talk",
        font="Times New Roman",
        rc={
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
            "axes.facecolor": PANEL_BG,
            "figure.facecolor": "#ffffff",
            "axes.edgecolor": BORDER,
            "grid.color": GRID,
            "axes.labelcolor": TEXT_DARK,
            "xtick.color": "#374151",
            "ytick.color": "#374151",
            "axes.titleweight": "bold",
            "axes.linewidth": 0.9,
            "grid.linewidth": 0.8,
        },
    )


def save_figure(fig: plt.Figure, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=320, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(output_path.with_suffix(".svg"), bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def crop_light_margin(image: np.ndarray, threshold: float = 0.985, pad: int = 12) -> np.ndarray:
    if image.ndim == 3:
        rgb = image[..., :3]
        mask = np.mean(rgb, axis=2) < threshold
    else:
        mask = image < threshold
    coords = np.argwhere(mask)
    if coords.size == 0:
        return image
    y0, x0 = coords.min(axis=0)[:2]
    y1, x1 = coords.max(axis=0)[:2] + 1
    y0 = max(0, y0 - pad)
    x0 = max(0, x0 - pad)
    y1 = min(image.shape[0], y1 + pad)
    x1 = min(image.shape[1], x1 + pad)
    return image[y0:y1, x0:x1]


def crop_fraction(image: np.ndarray, x0: float, y0: float, x1: float, y1: float) -> np.ndarray:
    h, w = image.shape[:2]
    return image[int(h * y0):int(h * y1), int(w * x0):int(w * x1)]


def load_snapshot(path: Path, crop: tuple[float, float, float, float] | None = None, auto_trim: bool = False) -> np.ndarray:
    img = mpimg.imread(path)
    if crop is not None:
        img = crop_fraction(img, *crop)
    if auto_trim:
        img = crop_light_margin(img)
    return img


def wrap_block_text(text: str, width: int) -> str:
    parts = []
    for para in str(text).split("\n"):
        para = para.strip()
        if not para:
            parts.append("")
            continue
        parts.append(textwrap.fill(para, width=width, break_long_words=False, break_on_hyphens=False))
    return "\n".join(parts)


def mechanism_card(
    fig: plt.Figure,
    box: tuple[float, float, float, float],
    title: str,
    body: str,
    accent: str,
) -> plt.Axes:
    ax = fig.add_axes(box)
    ax.set_facecolor("#ffffff")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor(BORDER)
        spine.set_linewidth(1.0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.add_patch(Rectangle((0, 0.80), 1, 0.20, facecolor=accent, edgecolor=accent, linewidth=0))
    ax.text(0.04, 0.90, title, fontsize=13, fontweight="bold", color="white", va="center", ha="left")
    ax.text(
        0.05,
        0.72,
        wrap_block_text(body, 34),
        fontsize=10.0,
        color=TEXT_DARK,
        va="top",
        ha="left",
        linespacing=1.38,
        clip_on=True,
    )
    return ax


def simple_stage_box(
    ax: plt.Axes,
    xy: tuple[float, float],
    wh: tuple[float, float],
    title: str,
    body: str,
    accent: str,
) -> None:
    x, y = xy
    w, h = wh
    ax.add_patch(Rectangle((x, y), w, h, facecolor="#ffffff", edgecolor=BORDER, linewidth=1.1))
    ax.add_patch(Rectangle((x, y + h - 0.12), w, 0.12, facecolor=accent, edgecolor=accent, linewidth=0))
    ax.text(x + 0.03, y + h - 0.06, title, fontsize=12.5, fontweight="bold", color="white", va="center", ha="left")
    ax.text(
        x + 0.03,
        y + h - 0.16,
        wrap_block_text(body, 20),
        fontsize=8.9,
        color=TEXT_DARK,
        va="top",
        ha="left",
        linespacing=1.30,
        clip_on=True,
    )


def add_arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float], color: str = "#4b5563", lw: float = 2.0) -> None:
    ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="-|>", color=color, linewidth=lw))


def get_contrast_label(contrast_df: pd.DataFrame, metric: str, contrast: str) -> str:
    subset = contrast_df[(contrast_df["metric"] == metric) & (contrast_df["contrast"] == contrast)]
    if subset.empty:
        return "n.s."
    row = subset.iloc[0]
    return f"{contrast}: p={row['p_value']:.3f}"


def add_panel_note(ax: plt.Axes, text: str, *, position: str = "top_right") -> None:
    if not text:
        return
    if position == "top_right":
        x, y, ha = 0.98, 0.90, "right"
    elif position == "top_left":
        x, y, ha = 0.02, 0.90, "left"
    elif position == "bottom_right":
        x, y, ha = 0.98, 0.04, "right"
    else:
        x, y, ha = 0.98, 0.06, "right"
    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        fontsize=9.2,
        color=TEXT_MUTED,
        ha=ha,
        va="bottom" if "bottom" in position else "top",
    )


def add_metric_panel(
    ax: plt.Axes,
    data: pd.DataFrame,
    x: str,
    y: str,
    order: list[str],
    title: str,
    ylabel: str,
    subtitle: str | None = None,
) -> None:
    rng = np.random.default_rng(7)
    for idx, condition in enumerate(order):
        subset = pd.to_numeric(data.loc[data[x] == condition, y], errors="coerce").dropna()
        if subset.empty:
            continue
        jitter = rng.normal(loc=idx, scale=0.04, size=len(subset))
        ax.scatter(
            jitter,
            subset,
            s=26,
            color=PALETTE[condition],
            alpha=0.28,
            edgecolor="white",
            linewidth=0.5,
            zorder=2,
        )
        mean = subset.mean()
        sem = subset.std(ddof=1) / np.sqrt(len(subset)) if len(subset) > 1 else 0
        ci = 1.96 * sem
        ax.vlines(idx, mean - ci, mean + ci, color=PALETTE[condition], linewidth=2.2, zorder=3)
        ax.hlines([mean - ci, mean + ci], idx - 0.05, idx + 0.05, color=PALETTE[condition], linewidth=1.3, zorder=3)
        ax.scatter(
            [idx],
            [mean],
            s=64,
            color=PALETTE[condition],
            edgecolor=TEXT_DARK,
            linewidth=0.6,
            zorder=4,
        )

    ax.set_xlim(-0.45, len(order) - 0.55)
    ax.set_xticks(range(len(order)), order)
    ax.set_title(title, loc="left", fontsize=14.5, pad=14, color=TEXT_DARK)
    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.grid(axis="x", visible=False)
    if subtitle:
        add_panel_note(ax, subtitle, position="bottom_right")


def build_behavioral_contrasts_figure(
    session_df: pd.DataFrame,
    session_contrast_df: pd.DataFrame,
    token_feature_df: pd.DataFrame,
    token_session_tests_df: pd.DataFrame,
    output_dir: Path,
) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(12.6, 8.8), facecolor="#ffffff")
    fig.subplots_adjust(top=0.96, bottom=0.08, left=0.08, right=0.98, hspace=0.32, wspace=0.28)

    add_metric_panel(
        axes[0, 0],
        session_df[session_df["condition"].isin(["Full", "No-Graph"])],
        "condition",
        "accepted_usable_units",
        ["Full", "No-Graph"],
        "A. Accepted usable output",
        "Accepted usable units",
        get_contrast_label(session_contrast_df, "accepted_usable_units", "Full vs No-Graph"),
    )
    add_metric_panel(
        axes[0, 1],
        token_feature_df[token_feature_df["condition"].isin(["Full", "No-Graph"])],
        "condition",
        "prompt_token_slope",
        ["Full", "No-Graph"],
        "B. Token escalation slope",
        "Prompt-token slope per ask round",
        get_contrast_label(token_session_tests_df, "prompt_token_slope", "Full vs No-Graph"),
    )
    add_metric_panel(
        axes[1, 0],
        token_feature_df[token_feature_df["condition"].isin(["Full", "No-Graph"])],
        "condition",
        "late_minus_early_prompt",
        ["Full", "No-Graph"],
        "C. Late minus early token load",
        "Late-stage minus early-stage tokens",
        get_contrast_label(token_session_tests_df, "late_minus_early_prompt", "Full vs No-Graph"),
    )
    add_metric_panel(
        axes[1, 1],
        session_df,
        "condition",
        "graph_block_count",
        ["Full", "No-Graph", "External"],
        "D. Graph carry-over availability",
        "Graph block count",
        get_contrast_label(session_contrast_df, "graph_block_count", "Full vs External"),
    )

    for ax in axes.flat:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(BORDER)
        ax.spines["bottom"].set_color(BORDER)
        ax.set_facecolor("#ffffff")

    output_path = output_dir / "fig01_behavioral_contrasts.png"
    save_figure(fig, output_path)
    return output_path


def build_token_round_trajectory_figure(
    rounds_df: pd.DataFrame,
    token_session_tests_df: pd.DataFrame,
    output_dir: Path,
) -> Path:
    data = rounds_df[rounds_df["condition"].isin(["Full", "No-Graph"])].copy()
    data["base_prompt_tokens"] = data.groupby("session_id")["prompt_tokens"].transform("first")
    data["relative_prompt_ratio"] = data["prompt_tokens"] / data["base_prompt_tokens"]
    data["round_bin"] = ((data["round_index"] - 1) // 2) + 1
    binned = (
        data.groupby(["session_id", "condition", "round_bin"], as_index=False)
        .agg(
            prompt_tokens=("prompt_tokens", "mean"),
            relative_prompt_ratio=("relative_prompt_ratio", "mean"),
        )
    )

    fig, axes = plt.subplots(1, 2, figsize=(13.6, 4.8), facecolor="#ffffff")
    fig.subplots_adjust(top=0.95, bottom=0.16, left=0.08, right=0.98, wspace=0.28)

    for ax, metric in zip(axes, ["prompt_tokens", "relative_prompt_ratio"]):
        for condition in ["Full", "No-Graph"]:
            subset = binned[binned["condition"] == condition]
            grouped = subset.groupby("round_bin")[metric].agg(["mean", "std", "count"]).reset_index()
            grouped["sem"] = grouped["std"].fillna(0) / np.sqrt(grouped["count"].clip(lower=1))
            frac = 0.45 if len(grouped) >= 5 else 0.8
            smooth_mean = lowess(grouped["mean"], grouped["round_bin"], frac=frac, return_sorted=False)
            lower = lowess(grouped["mean"] - grouped["sem"], grouped["round_bin"], frac=frac, return_sorted=False)
            upper = lowess(grouped["mean"] + grouped["sem"], grouped["round_bin"], frac=frac, return_sorted=False)
            ax.fill_between(grouped["round_bin"], lower, upper, color=PALETTE[condition], alpha=0.12, linewidth=0)
            ax.plot(grouped["round_bin"], smooth_mean, color=PALETTE[condition], linewidth=2.4, label=condition)
            ax.scatter(grouped["round_bin"], grouped["mean"], s=22, color=PALETTE[condition], alpha=0.9, zorder=3)

    axes[0].set_title("A. Raw prompt-token load", loc="left", fontsize=14.5, pad=12)
    axes[0].set_xlabel("Ask-round bin (2 rounds -> 1 bin)")
    axes[0].set_ylabel("Prompt tokens")
    add_panel_note(
        axes[0],
        get_contrast_label(token_session_tests_df, "prompt_token_slope", "Full vs No-Graph"),
        position="top_right",
    )

    axes[1].set_title("B. Prompt growth normalized to the first ask", loc="left", fontsize=14.5, pad=12)
    axes[1].set_xlabel("Ask-round bin (2 rounds -> 1 bin)")
    axes[1].set_ylabel("Prompt tokens / first-round prompt")
    add_panel_note(
        axes[1],
        get_contrast_label(token_session_tests_df, "late_over_early_prompt_ratio", "Full vs No-Graph"),
        position="top_right",
    )

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(BORDER)
        ax.spines["bottom"].set_color(BORDER)
        ax.set_facecolor("#ffffff")
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.grid(axis="x", visible=False)
    axes[0].legend(frameon=False, loc="upper left", ncol=2, handlelength=1.8, columnspacing=1.1)
    if axes[1].legend_:
        axes[1].legend_.remove()

    output_path = output_dir / "fig02_token_round_trajectories.png"
    save_figure(fig, output_path)
    return output_path


def build_phase_dynamics_figure(phase_df: pd.DataFrame, output_dir: Path) -> Path:
    data = phase_df[phase_df["condition"].isin(["Full", "No-Graph"])].copy()
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.8), facecolor="#ffffff")
    fig.subplots_adjust(top=0.95, bottom=0.18, left=0.07, right=0.98, wspace=0.30)

    metrics = [
        ("prompt_tokens", "A. Prompt tokens by phase", "Prompt tokens"),
        ("invoke_count", "B. Ask frequency by phase", "Completed asks"),
        ("accepted_usable_units", "C. Accepted output by phase", "Accepted usable units"),
    ]
    for ax, (metric, title, ylabel) in zip(axes, metrics):
        sns.lineplot(
            data=data,
            x="phase_number",
            y=metric,
            hue="condition",
            hue_order=["Full", "No-Graph"],
            palette=PALETTE,
            marker="o",
            linewidth=3,
            errorbar=("ci", 95),
            ax=ax,
        )
        ax.set_title(title, loc="left", fontsize=14.5, pad=12)
        ax.set_xlabel("Phase")
        ax.set_ylabel(ylabel)
        ax.set_xticks([1, 2, 3], ["P1", "P2", "P3"])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(BORDER)
        ax.spines["bottom"].set_color(BORDER)
        ax.set_facecolor("#ffffff")
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.grid(axis="x", visible=False)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles[:2], labels[:2], frameon=False, loc="upper right", fontsize=10)

    output_path = output_dir / "fig03_phase_dynamics.png"
    save_figure(fig, output_path)
    return output_path


def build_external_imputation_figure(imputed_table_df: pd.DataFrame, output_dir: Path) -> Path:
    melted = imputed_table_df.melt(
        id_vars=["metric", "note"],
        value_vars=["Full", "No-Graph", "External"],
        var_name="condition",
        value_name="value",
    )
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.8), facecolor="#ffffff")
    fig.subplots_adjust(top=0.95, bottom=0.15, left=0.06, right=0.98, wspace=0.32)

    metrics = [
        "AI Invoke Times (Filled)",
        "Straight-Use Rate (Filled)",
        "Rewrite Ratio (Filled)",
    ]
    for ax, metric in zip(axes, metrics):
        subset = melted[melted["metric"] == metric].copy()
        sns.barplot(
            data=subset,
            x="condition",
            y="value",
            hue="condition",
            order=["Full", "No-Graph", "External"],
            palette=PALETTE,
            legend=False,
            ax=ax,
        )
        ax.set_title(metric.replace(" (Filled)", ""), loc="left", fontsize=14.5, pad=12)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(BORDER)
        ax.spines["bottom"].set_color(BORDER)
        ax.set_facecolor("#ffffff")
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.grid(axis="x", visible=False)
        for patch, cond in zip(ax.patches, ["Full", "No-Graph", "External"]):
            if cond == "External":
                patch.set_hatch("///")
                patch.set_edgecolor("#111827")
        add_panel_note(ax, "Hatched bar = imputed External", position="top_right")

    output_path = output_dir / "fig04_external_imputation.png"
    save_figure(fig, output_path)
    return output_path


def build_artifact_examples_figure(experiments_dir: Path, session_raw_df: pd.DataFrame, output_dir: Path) -> Path:
    preview_specs = [
        (
            "A. AIPad-Full artifact",
            experiments_dir / "snapshot" / "B-full.png",
            (0.05, 0.10, 0.96, 0.86),
        ),
        (
            "B. AIPad-NoGraph artifact",
            experiments_dir / "snapshot" / "屏幕截图 2026-03-17 205515.png",
            (0.06, 0.10, 0.97, 0.86),
        ),
        (
            "C. External-chat note artifact",
            experiments_dir / "snapshot" / "C-external.png",
            (0.07, 0.12, 0.96, 0.86),
        ),
    ]
    quality_counts = (
        session_raw_df["quality_flags"]
        .replace({"ok": "clean"})
        .value_counts()
        .rename_axis("issue_group")
        .reset_index(name="count")
    )

    fig = plt.figure(figsize=(12.8, 8.6), facecolor="#ffffff")
    grid = fig.add_gridspec(2, 2, hspace=0.20, wspace=0.16)
    axes = [
        fig.add_subplot(grid[0, 0]),
        fig.add_subplot(grid[0, 1]),
        fig.add_subplot(grid[1, 0]),
        fig.add_subplot(grid[1, 1]),
    ]
    fig.subplots_adjust(top=0.96, bottom=0.08, left=0.04, right=0.98)

    for ax, (title, image_path, crop_box) in zip(axes[:3], preview_specs):
        img = load_snapshot(image_path, crop=crop_box, auto_trim=True)
        ax.imshow(img, aspect="auto")
        ax.set_title(title, loc="left", fontsize=13.8, pad=8, color=TEXT_DARK)
        ax.axis("off")
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor(BORDER)
            spine.set_linewidth(1.0)
        ax.set_facecolor("#ffffff")

    failure_ax = axes[3]
    failure_ax.set_facecolor("#ffffff")
    sns.barplot(
        data=quality_counts.head(4),
        x="count",
        y="issue_group",
        color="#b45309",
        ax=failure_ax,
    )
    failure_ax.set_title("D. Failure / repair cases in the log corpus", loc="left", fontsize=13.8, pad=8, color=TEXT_DARK)
    failure_ax.set_xlabel("Session count")
    failure_ax.set_ylabel("")
    failure_ax.spines["top"].set_visible(False)
    failure_ax.spines["right"].set_visible(False)
    failure_ax.spines["left"].set_color(BORDER)
    failure_ax.spines["bottom"].set_color(BORDER)
    failure_ax.grid(axis="x", color=GRID, linewidth=0.8)
    failure_ax.grid(axis="y", visible=False)
    failure_ax.add_patch(
        Rectangle(
            (0.02, 0.02),
            0.96,
            0.25,
            transform=failure_ax.transAxes,
            facecolor="#ffffff",
            edgecolor=BORDER,
            linewidth=0.9,
        )
    )
    failure_ax.text(
        0.04,
        0.235,
        "Representative issues in the repaired corpus:\n"
        "1. Missing end markers repaired from export timestamps\n"
        "2. Phase logging gaps repaired from protocol windows\n"
        "3. One External session had internal AI contamination removed",
        transform=failure_ax.transAxes,
        fontsize=9.8,
        color=TEXT_DARK,
        va="top",
    )
    failure_ax.tick_params(axis="y", labelsize=9.8)
    failure_ax.tick_params(axis="x", labelsize=9.8)

    output_path = output_dir / "fig05_artifact_examples_and_failures.png"
    save_figure(fig, output_path)
    return output_path


def build_mechanism_overview_figure(experiments_dir: Path, output_dir: Path) -> Path:
    base = load_snapshot(experiments_dir / "snapshot" / "P1.png", crop=(0.01, 0.03, 0.99, 0.97))
    ext = load_snapshot(experiments_dir / "snapshot" / "C-external.png", crop=(0.48, 0.02, 0.98, 0.48))

    fig = plt.figure(figsize=(16.5, 8.5), facecolor="#ffffff")

    title_ax = fig.add_axes([0.03, 0.79, 0.22, 0.17])
    title_ax.axis("off")
    title_ax.text(0.0, 0.92, "AIPad:", fontsize=28, fontweight="bold", color=TEXT_DARK, ha="left", va="top")
    title_ax.text(
        0.0,
        0.60,
        "In-Place AI Collaboration and\nWorkspace-Carried Context for\nLearning Notes",
        fontsize=18,
        color="#30343f",
        style="italic",
        ha="left",
        va="top",
        linespacing=1.25,
    )

    main_ax = fig.add_axes([0.23, 0.18, 0.54, 0.70])
    main_ax.imshow(base)
    main_ax.axis("off")
    for spine in main_ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor(BORDER)
        spine.set_linewidth(1.0)

    inset_ax = fig.add_axes([0.74, 0.58, 0.23, 0.30])
    inset_ax.imshow(ext)
    inset_ax.axis("off")
    for spine in inset_ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor(BORDER)
        spine.set_linewidth(1.0)
    inset_ax.text(
        0.98,
        0.02,
        "No more split-workflow",
        fontsize=12,
        fontweight="bold",
        color="white",
        ha="right",
        va="bottom",
        bbox={"facecolor": "#111827", "edgecolor": "#111827", "pad": 4},
        transform=inset_ax.transAxes,
    )

    card1 = mechanism_card(
        fig,
        (0.03, 0.41, 0.20, 0.27),
        "Context-Carried Orchestration",
        "Collaborate without restatement.\nThe Orchestrator compiles locally relevant context from the evolving workspace, so the AI can continue work without repeated background handoff.",
        "#c4511c",
    )
    card2 = mechanism_card(
        fig,
        (0.63, 0.12, 0.33, 0.28),
        "AutoMaintain: Semantic Workspace Memory",
        "Moving beyond raw history. The system incrementally structures fragments into semantic blocks, maintaining a persistent memory of concepts and relations.",
        "#395b9a",
    )
    card3 = mechanism_card(
        fig,
        (0.04, 0.07, 0.38, 0.16),
        "Preview-First In-Place Interaction",
        "Preserve the flow of thought. AI suggestions appear directly within the workspace, allowing seamless acceptance or revision without context switching.",
        "#2f8f46",
    )

    overlay = fig.add_axes([0, 0, 1, 1], frameon=False)
    overlay.set_xlim(0, 1)
    overlay.set_ylim(0, 1)
    overlay.axis("off")
    overlay.annotate("", xy=(0.30, 0.57), xytext=(0.21, 0.54), arrowprops=dict(arrowstyle="-", color="#c4511c", linewidth=3))
    overlay.scatter([0.21, 0.30], [0.54, 0.57], color="#c4511c", s=110, zorder=5)
    overlay.annotate("", xy=(0.68, 0.64), xytext=(0.57, 0.64), arrowprops=dict(arrowstyle="-", color="#395b9a", linewidth=3))
    overlay.annotate("", xy=(0.66, 0.48), xytext=(0.55, 0.48), arrowprops=dict(arrowstyle="-", color="#395b9a", linewidth=3))
    overlay.scatter([0.57, 0.55, 0.68, 0.66], [0.64, 0.48, 0.64, 0.48], color="#395b9a", s=110, zorder=5)
    overlay.annotate("", xy=(0.44, 0.27), xytext=(0.27, 0.15), arrowprops=dict(arrowstyle="-", color="#2f8f46", linewidth=3))
    overlay.scatter([0.27, 0.44], [0.15, 0.27], color="#2f8f46", s=110, zorder=5)

    output_path = output_dir / "fig06_mechanism_overview.png"
    save_figure(fig, output_path)
    return output_path


def build_preview_interaction_mechanism(experiments_dir: Path, output_dir: Path) -> Path:
    img = load_snapshot(experiments_dir / "snapshot" / "屏幕截图 2026-03-17 205515.png", crop=(0.02, 0.02, 0.98, 0.96))
    fig = plt.figure(figsize=(14.2, 7.2), facecolor="#ffffff")
    main_ax = fig.add_axes([0.04, 0.10, 0.70, 0.82])
    main_ax.imshow(img)
    main_ax.axis("off")

    card = mechanism_card(
        fig,
        (0.76, 0.53, 0.20, 0.31),
        "Interaction Contract",
        "1. Ask AI from the canvas.\n2. Inspect the in-place preview.\n3. Accept or dismiss before it becomes part of the note.\nThis keeps AI intervention frequent without giving up authorship control.",
        "#2f8f46",
    )
    card2 = mechanism_card(
        fig,
        (0.76, 0.17, 0.20, 0.24),
        "Observed Benefit",
        "The in-canvas conditions achieved higher calibrated straight-use and lower rewrite burden than the external workflow, showing that preview-first interaction lowers the cost of recruiting AI into ongoing work.",
        "#395b9a",
    )

    overlay = fig.add_axes([0, 0, 1, 1], frameon=False)
    overlay.set_xlim(0, 1)
    overlay.set_ylim(0, 1)
    overlay.axis("off")
    overlay.annotate("", xy=(0.54, 0.90), xytext=(0.76, 0.72), arrowprops=dict(arrowstyle="-|>", color="#2f8f46", linewidth=2.4))
    overlay.annotate("", xy=(0.59, 0.90), xytext=(0.76, 0.65), arrowprops=dict(arrowstyle="-|>", color="#2f8f46", linewidth=2.4))
    overlay.annotate("", xy=(0.66, 0.18), xytext=(0.76, 0.28), arrowprops=dict(arrowstyle="-|>", color="#395b9a", linewidth=2.4))

    output_path = output_dir / "fig07_preview_interaction_mechanism.png"
    save_figure(fig, output_path)
    return output_path


def build_workspace_memory_mechanism(experiments_dir: Path, output_dir: Path) -> Path:
    full_img = load_snapshot(experiments_dir / "snapshot" / "P1.png", crop=(0.22, 0.04, 0.98, 0.96))
    ext_img = load_snapshot(experiments_dir / "snapshot" / "C-external.png", crop=(0.52, 0.06, 0.98, 0.62))
    fig = plt.figure(figsize=(14.6, 7.6), facecolor="#ffffff")

    left = fig.add_axes([0.04, 0.14, 0.58, 0.76])
    left.imshow(full_img)
    left.axis("off")
    left.text(0.0, 1.02, "A. Full: maintained workspace memory", fontsize=14, fontweight="bold", color=TEXT_DARK, transform=left.transAxes)

    right_top = fig.add_axes([0.67, 0.54, 0.28, 0.30])
    right_top.imshow(ext_img)
    right_top.axis("off")
    right_top.text(0.0, 1.02, "B. External: no carried canvas memory", fontsize=13, fontweight="bold", color=TEXT_DARK, transform=right_top.transAxes)

    note = mechanism_card(
        fig,
        (0.67, 0.16, 0.29, 0.26),
        "Why Memory Matters",
        "In Full, graph-maintained blocks and planner focus keep later requests anchored to the evolving page structure. Without this layer, later help may remain broad but has to be re-fit manually to the note.",
        "#395b9a",
    )

    overlay = fig.add_axes([0, 0, 1, 1], frameon=False)
    overlay.set_xlim(0, 1)
    overlay.set_ylim(0, 1)
    overlay.axis("off")
    overlay.annotate("", xy=(0.54, 0.71), xytext=(0.67, 0.28), arrowprops=dict(arrowstyle="-|>", color="#395b9a", linewidth=2.6))
    overlay.annotate("", xy=(0.57, 0.18), xytext=(0.67, 0.26), arrowprops=dict(arrowstyle="-|>", color="#395b9a", linewidth=2.6))

    output_path = output_dir / "fig08_workspace_memory_mechanism.png"
    save_figure(fig, output_path)
    return output_path


def build_automaintain_pipeline_diagram(output_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(15.2, 5.8), facecolor="#ffffff")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    simple_stage_box(
        ax, (0.03, 0.22), (0.18, 0.56),
        "1. Canvas Traces",
        "User strokes, text boxes, and accepted AI drafts are first collected as protocol-level canvas traces.\n\nKey modules:\n`LineArtBoard.tsx`\n`ai/plan.ts`",
        "#334155",
    )
    simple_stage_box(
        ax, (0.27, 0.18), (0.18, 0.64),
        "2. Fragmentization",
        "Backend sync converts accepted strokes into semantic fragments.\n\nText fragments can match existing blocks semantically; stroke fragments first enter pending vision flow.",
        "#0f766e",
    )
    simple_stage_box(
        ax, (0.51, 0.18), (0.18, 0.64),
        "3. Group -> Block",
        "Pending groups accumulate until stable enough to promote. Stable groups become semantic blocks with labels, summaries, and relations.\n\nCore path:\n`fragment -> group -> block`",
        "#2563eb",
    )
    simple_stage_box(
        ax, (0.75, 0.22), (0.22, 0.56),
        "4. Reusable Workspace Memory",
        "Promoted blocks, relations, and recency metadata become the memory layer used by later requests.\n\nThis is the structure that carries context forward across long note sessions.",
        "#395b9a",
    )

    add_arrow(ax, (0.21, 0.50), (0.27, 0.50), color="#64748b", lw=2.2)
    add_arrow(ax, (0.45, 0.50), (0.51, 0.50), color="#64748b", lw=2.2)
    add_arrow(ax, (0.69, 0.50), (0.75, 0.50), color="#64748b", lw=2.2)

    ax.text(0.50, 0.92, "AutoMaintain Pipeline: From Canvas Traces To Reusable Semantic Memory", fontsize=18, fontweight="bold", ha="center", color=TEXT_DARK)
    ax.text(0.50, 0.07, "Based on README + workflow: strokes are ingested as fragments, clustered into groups, promoted into blocks, then reused by later requests.", fontsize=10.5, ha="center", color=TEXT_MUTED)

    output_path = output_dir / "fig09_automaintain_pipeline.png"
    save_figure(fig, output_path)
    return output_path


def build_context_routing_diagram(output_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(15.2, 6.4), facecolor="#ffffff")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    simple_stage_box(
        ax, (0.04, 0.18), (0.22, 0.68),
        "Context Sources",
        "1. `plan.targetBlockIds`\n2. `active_block_ids`\n3. `main_block_id`\n4. recently updated blocks\n5. relationship expansion",
        "#334155",
    )
    simple_stage_box(
        ax, (0.34, 0.18), (0.22, 0.68),
        "Planner",
        "The planner sees user hint, active blocks, primary block, related structures, and recent updates.\n\nOutputs:\n`action`\n`targetBlockIds`\n`nextStepHint`",
        "#0f766e",
    )
    simple_stage_box(
        ax, (0.63, 0.18), (0.16, 0.68),
        "Executor",
        "The executor turns planner output into a compact `block_outline`, prioritizing target blocks and keeping related blocks summary-only to reduce drift.",
        "#2563eb",
    )
    simple_stage_box(
        ax, (0.84, 0.25), (0.13, 0.54),
        "FULL / LIGHT Prompt",
        "Selected blocks + hint + block outline become the final prompt context.\n\nGoal: preserve local alignment without prompt overload.",
        "#395b9a",
    )

    add_arrow(ax, (0.26, 0.52), (0.34, 0.52), color="#64748b", lw=2.2)
    add_arrow(ax, (0.56, 0.52), (0.63, 0.52), color="#64748b", lw=2.2)
    add_arrow(ax, (0.79, 0.52), (0.84, 0.52), color="#64748b", lw=2.2)
    ax.text(0.50, 0.94, "Planner / Executor Context Routing", fontsize=18, fontweight="bold", ha="center", color=TEXT_DARK)
    ax.text(0.50, 0.08, "This diagram matches the current executor policy in README/workflow: route requests through maintained semantic structure instead of passing raw history wholesale.", fontsize=10.2, ha="center", color=TEXT_MUTED)

    output_path = output_dir / "fig10_context_routing.png"
    save_figure(fig, output_path)
    return output_path


def build_preview_accept_loop_diagram(output_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(14.5, 5.8), facecolor="#ffffff")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    simple_stage_box(
        ax, (0.04, 0.22), (0.18, 0.56),
        "Ask AI",
        "The frontend packages `sid`, delta, current draw stack, mode, and hint into a suggestion request.",
        "#334155",
    )
    simple_stage_box(
        ax, (0.29, 0.22), (0.18, 0.56),
        "Preview Layer",
        "Returned payload is normalized, planned into drafts, and staged as an in-place preview rather than committed output.",
        "#2f8f46",
    )
    simple_stage_box(
        ax, (0.54, 0.22), (0.18, 0.56),
        "Accept / Dismiss",
        "Accept writes drafts into shapes + drawStack.\nDismiss clears preview only.\nThe page remains under user control.",
        "#c4511c",
    )
    simple_stage_box(
        ax, (0.79, 0.22), (0.17, 0.56),
        "Re-ingest",
        "Accepted output re-enters sync and AutoMaintain, so AI output can become future workspace memory.",
        "#395b9a",
    )

    add_arrow(ax, (0.22, 0.50), (0.29, 0.50), color="#64748b", lw=2.2)
    add_arrow(ax, (0.47, 0.50), (0.54, 0.50), color="#64748b", lw=2.2)
    add_arrow(ax, (0.72, 0.50), (0.79, 0.50), color="#64748b", lw=2.2)
    add_arrow(ax, (0.87, 0.22), (0.13, 0.18), color="#94a3b8", lw=1.8)

    ax.text(0.50, 0.92, "Preview-First Interaction Loop", fontsize=18, fontweight="bold", ha="center", color=TEXT_DARK)
    ax.text(0.50, 0.08, "This is the core interaction contract in the codebase: AI suggestions arrive as reversible in-place previews, then re-enter the semantic runtime only after acceptance.", fontsize=10.3, ha="center", color=TEXT_MUTED)

    output_path = output_dir / "fig11_preview_accept_loop.png"
    save_figure(fig, output_path)
    return output_path


def build_figure_index(paths: list[tuple[str, Path, str]], output_dir: Path) -> Path:
    lines = ["# Figure Index", ""]
    for title, path, usage in paths:
        lines.append(f"- `{path.name}`: {title}")
        lines.append(f"  Suggested use: {usage}")
    output_path = output_dir / "figure_index.md"
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def main() -> None:
    args = parse_args()
    set_plot_style()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    analysis_xl = pd.ExcelFile(args.analysis_xlsx)
    experiment_xl = pd.ExcelFile(args.experiment_xlsx)

    session_df = experiment_xl.parse("session_summary")
    session_raw_df = experiment_xl.parse("session_summary_raw")
    phase_df = experiment_xl.parse("phase_metrics")
    session_contrast_df = analysis_xl.parse("session_contrasts")
    token_session_tests_df = analysis_xl.parse("token_session_tests")
    token_feature_df = analysis_xl.parse("token_escalation")
    imputed_table_df = analysis_xl.parse("paper_table_filled")
    rounds_df = load_completed_rounds(args.experiments_dir)

    paths = [
        (
            "Behavioral contrasts that best support the mechanism story",
            build_behavioral_contrasts_figure(
                session_df=session_df,
                session_contrast_df=session_contrast_df,
                token_feature_df=token_feature_df,
                token_session_tests_df=token_session_tests_df,
                output_dir=args.output_dir,
            ),
            "Main behavioral figure for Results; can replace a generic participant-level contrast placeholder.",
        ),
        (
            "Prompt-token trajectories over ask rounds",
            build_token_round_trajectory_figure(
                rounds_df=rounds_df,
                token_session_tests_df=token_session_tests_df,
                output_dir=args.output_dir,
            ),
            "Directly supports the Full vs No-Graph token-escalation argument.",
        ),
        (
            "Phase-by-phase dynamics for Full vs No-Graph",
            build_phase_dynamics_figure(phase_df=phase_df, output_dir=args.output_dir),
            "Supports the process explanation for why token burden diverges over time.",
        ),
        (
            "External imputation comparison bars",
            build_external_imputation_figure(imputed_table_df=imputed_table_df, output_dir=args.output_dir),
            "Use only when you explicitly disclose the External imputation assumptions.",
        ),
        (
            "Artifact examples and logging failure context",
            build_artifact_examples_figure(
                experiments_dir=args.experiments_dir,
                session_raw_df=session_raw_df,
                output_dir=args.output_dir,
            ),
            "Closest current replacement for the Figure 5 placeholder.",
        ),
    ]
    index_path = build_figure_index(paths, args.output_dir)

    print(f"Wrote {len(paths)} figure files to {args.output_dir}")
    print(f"Wrote figure index to {index_path}")


if __name__ == "__main__":
    main()
