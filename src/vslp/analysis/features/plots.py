from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NAVY = "#071A33"
TEAL = "#2DB7B0"
GOLD = "#B68B2D"
RED = "#B42318"
MUTED = "#607089"
GRID = "#D9E2EF"


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def _empty(path: Path, message: str, title: str = "VSLP Feature Analysis") -> Path:
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.text(0.5, 0.54, title, ha="center", va="center", fontsize=16, fontweight="bold", color=NAVY)
    ax.text(0.5, 0.42, message, ha="center", va="center", fontsize=11, color=MUTED, wrap=True)
    ax.axis("off")
    return _save(fig, path)


def _style(ax, title: str | None = None) -> None:
    ax.set_facecolor("white")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="x", color=GRID, alpha=0.55, linewidth=0.8)
    if title:
        ax.set_title(title, fontsize=14, fontweight="bold", color=NAVY, pad=12)


def plot_missingness(summary: pd.DataFrame, path: Path, top_n: int = 30) -> Path:
    if summary is None or summary.empty or "missing_fraction" not in summary.columns:
        return _empty(path, "No missingness data were available.", "Missingness")
    s = summary.copy()
    if "feature" not in s.columns:
        # Fallback: use index as feature name.
        s["feature"] = s.index.astype(str)
    s["missing_fraction"] = pd.to_numeric(s["missing_fraction"], errors="coerce")
    s = s.dropna(subset=["missing_fraction"]).sort_values("missing_fraction", ascending=False).head(top_n)
    if s.empty:
        return _empty(path, "No feature missingness values were available.", "Missingness")
    fig, ax = plt.subplots(figsize=(10, max(4.8, 0.32 * len(s))))
    colors = [RED if v >= 0.40 else GOLD if v >= 0.15 else TEAL for v in s["missing_fraction"]]
    ax.barh(s["feature"].astype(str), s["missing_fraction"], color=colors)
    ax.invert_yaxis()
    ax.set_xlabel("Missing fraction", color=MUTED)
    ax.set_xlim(0, 1)
    _style(ax, "Top missing feature columns")
    ax.axvline(0.15, color=GOLD, linestyle="--", linewidth=1, alpha=0.8)
    ax.axvline(0.40, color=RED, linestyle="--", linewidth=1, alpha=0.8)
    ax.text(0.15, -0.8, "15%", color=GOLD, fontsize=8)
    ax.text(0.40, -0.8, "40%", color=RED, fontsize=8)
    return _save(fig, path)


def plot_feature_availability_heatmap(df: pd.DataFrame, feature_cols: Sequence[str], path: Path, max_features: int | None = None, max_rows: int | None = None) -> Path:
    # Prefer full-dataset display when feasible. Earlier versions capped this plot
    # aggressively, which could hide task- or feature-specific missingness blocks.
    cols_all = list(feature_cols)
    cols = cols_all if max_features is None else cols_all[:max_features]
    if df is None or df.empty or not cols:
        return _empty(path, "No feature columns were detected for the availability matrix.", "Feature availability")
    show = df.loc[:, cols] if max_rows is None else df.loc[:, cols].head(max_rows)
    mat = (~show.isna()).astype(int).to_numpy()
    fig, ax = plt.subplots(figsize=(12, max(5, min(12, 0.055 * len(show) + 3.5))))
    im = ax.imshow(mat, aspect="auto", interpolation="nearest", cmap="viridis", vmin=0, vmax=1)
    ax.set_title("Feature availability matrix", fontsize=14, fontweight="bold", color=NAVY, pad=12)
    ax.set_xlabel(f"Feature columns shown: {len(cols)} / {len(feature_cols)}", color=MUTED)
    ax.set_ylabel(f"Rows shown: {len(show)} / {len(df)}", color=MUTED)
    tick_step = max(1, len(cols) // 35)
    xticks = list(range(0, len(cols), tick_step))
    ax.set_xticks(xticks)
    ax.set_xticklabels([cols[i] for i in xticks], rotation=90, fontsize=6, color=MUTED)
    ax.set_yticks([])
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_ticks([0, 1]); cbar.set_ticklabels(["missing", "available"])
    cbar.ax.tick_params(labelsize=8, colors=MUTED)
    return _save(fig, path)


def plot_feature_availability_bars(summary: pd.DataFrame, path: Path, top_n: int = 30) -> Path:
    """Readable availability plot replacing dense feature x row heatmaps.

    Expects a feature-level missingness table from missingness_feature_summary.
    Plots available and missing row counts for the features with the lowest
    availability in the current task/context scope.
    """
    if summary is None or summary.empty:
        return _empty(path, "No feature availability data were available for the current scope.", "Feature availability")
    df = summary.copy()
    if "feature" not in df.columns:
        df["feature"] = df.index.astype(str)
    if "missing_fraction" not in df.columns:
        return _empty(path, "Feature missingness fractions were not available for this scope.", "Feature availability")
    df["missing_fraction"] = pd.to_numeric(df["missing_fraction"], errors="coerce")
    if "n_rows" in df.columns:
        df["n_rows"] = pd.to_numeric(df["n_rows"], errors="coerce")
    elif "n" in df.columns:
        df["n_rows"] = pd.to_numeric(df["n"], errors="coerce")
    else:
        # Recover a useful denominator when only fractions are present.
        df["n_rows"] = np.nan
    if "n_missing" in df.columns:
        df["missing_rows"] = pd.to_numeric(df["n_missing"], errors="coerce")
    elif "missing_count" in df.columns:
        df["missing_rows"] = pd.to_numeric(df["missing_count"], errors="coerce")
    else:
        df["missing_rows"] = df["missing_fraction"] * df["n_rows"]
    if "n_available" in df.columns:
        df["available_rows"] = pd.to_numeric(df["n_available"], errors="coerce")
    elif "available_count" in df.columns:
        df["available_rows"] = pd.to_numeric(df["available_count"], errors="coerce")
    else:
        df["available_rows"] = df["n_rows"] - df["missing_rows"]
    df = df.dropna(subset=["missing_fraction"]).sort_values("missing_fraction", ascending=False).head(top_n)
    if df.empty:
        return _empty(path, "No feature availability values were available for this scope.", "Feature availability")
    # If counts could not be recovered, fall back to percentages.
    if df[["available_rows", "missing_rows"]].isna().all().all():
        df["available_rows"] = 1.0 - df["missing_fraction"]
        df["missing_rows"] = df["missing_fraction"]
        xlabel = "Fraction of rows"
    else:
        df["available_rows"] = df["available_rows"].fillna(0)
        df["missing_rows"] = df["missing_rows"].fillna(0)
        xlabel = "Rows / recordings"
    labels = df["feature"].astype(str)
    y = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(11, max(5, 0.36 * len(df))))
    ax.barh(y, df["available_rows"].astype(float), color=TEAL, label="available")
    ax.barh(y, df["missing_rows"].astype(float), left=df["available_rows"].astype(float), color=RED, label="missing")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8, color=MUTED)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel, color=MUTED)
    _style(ax, "Feature availability summary")
    ax.legend(frameon=False, fontsize=8)
    return _save(fig, path)


def plot_comissing_pair_bars(pairs: pd.DataFrame, path: Path, top_n: int = 30) -> Path:
    """Readable co-missingness plot replacing feature x feature heatmaps."""
    if pairs is None or pairs.empty:
        return _empty(path, "No co-missing feature pairs were available for the current scope.", "Co-missingness pairs")
    df = pairs.copy()
    # Support several possible audit table schemas.
    if "feature_pair" not in df.columns:
        if {"feature_a", "feature_b"}.issubset(df.columns):
            df["feature_pair"] = df["feature_a"].astype(str) + " + " + df["feature_b"].astype(str)
        elif {"feature_1", "feature_2"}.issubset(df.columns):
            df["feature_pair"] = df["feature_1"].astype(str) + " + " + df["feature_2"].astype(str)
        else:
            df["feature_pair"] = df.index.astype(str)
    value_col = next((c for c in ["co_missing_fraction", "pair_missing_fraction", "fraction_co_missing", "co_missing_rate", "n_co_missing"] if c in df.columns), None)
    if value_col is None:
        return _empty(path, "Co-missingness scores were not available for this scope.", "Co-missingness pairs")
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df = df.dropna(subset=[value_col]).sort_values(value_col, ascending=False).head(top_n)
    if df.empty:
        return _empty(path, "No usable co-missingness scores were available for this scope.", "Co-missingness pairs")
    df = df.sort_values(value_col, ascending=True)
    fig, ax = plt.subplots(figsize=(11, max(5, 0.36 * len(df))))
    colors = [RED if v >= 0.50 else GOLD if v >= 0.20 else TEAL for v in df[value_col].astype(float)]
    ax.barh(df["feature_pair"].astype(str), df[value_col].astype(float), color=colors)
    ax.set_xlabel("Co-missing fraction" if "fraction" in value_col or "rate" in value_col else "Co-missing rows", color=MUTED)
    _style(ax, "Top co-missing feature pairs")
    return _save(fig, path)

def plot_distribution_grid(df: pd.DataFrame, feature_cols: Sequence[str], path: Path, max_features: int = 12) -> Path:
    cols = list(feature_cols)[:max_features]
    if df is None or df.empty or not cols:
        return _empty(path, "No numeric features were available for distribution plotting.", "Feature distributions")
    n = len(cols); ncols = 3; nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 3.4 * nrows))
    axes = np.array(axes).reshape(-1)
    for ax, col in zip(axes, cols):
        x = pd.to_numeric(df[col], errors="coerce").dropna()
        if x.empty:
            ax.text(.5, .5, "all missing", ha="center", va="center", color=MUTED)
            ax.set_title(col, fontsize=9, color=NAVY)
            ax.axis("off")
            continue
        ax.hist(x, bins=min(24, max(6, int(np.sqrt(len(x))))), color=TEAL, alpha=0.88)
        med = x.median(); ax.axvline(med, color=NAVY, linestyle="--", linewidth=1)
        ax.set_title(col, fontsize=9, color=NAVY)
        ax.set_ylabel("count", color=MUTED)
        _style(ax)
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle("Feature distributions", fontsize=15, fontweight="bold", color=NAVY, y=1.01)
    return _save(fig, path)


def plot_corr_heatmap(corr: pd.DataFrame, path: Path, title: str = "Spearman correlation heatmap") -> Path:
    if corr is None or corr.empty:
        return _empty(path, "Not enough numeric features were available for correlation analysis.", title)
    fig, ax = plt.subplots(figsize=(max(8, 0.26 * len(corr.columns)), max(7, 0.26 * len(corr.index))))
    im = ax.imshow(corr.to_numpy(dtype=float), vmin=-1, vmax=1, aspect="auto", cmap="coolwarm")
    ax.set_title(title, fontsize=14, fontweight="bold", color=NAVY, pad=12)
    ax.set_xticks(range(len(corr.columns))); ax.set_xticklabels(corr.columns, rotation=90, fontsize=6, color=MUTED)
    ax.set_yticks(range(len(corr.index))); ax.set_yticklabels(corr.index, fontsize=6, color=MUTED)
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Spearman rho", color=MUTED)
    cbar.ax.tick_params(labelsize=8, colors=MUTED)
    return _save(fig, path)


def plot_outlier_counts(outliers: pd.DataFrame, path: Path) -> Path:
    if outliers is None or outliers.empty or "feature" not in outliers.columns:
        return _empty(path, "No robust outliers were detected or no outlier table was available.", "Robust outliers")
    counts = outliers["feature"].value_counts().head(30).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(10, max(4.8, 0.32 * len(counts))))
    ax.barh(counts.index.astype(str), counts.values, color=GOLD)
    ax.set_xlabel("Outlier count", color=MUTED)
    _style(ax, "Robust outliers by feature")
    return _save(fig, path)


def plot_role_counts(role_summary: pd.DataFrame, path: Path) -> Path:
    if role_summary is None or role_summary.empty or "role" not in role_summary.columns or "n_columns" not in role_summary.columns:
        return _empty(path, "No column-role summary was available.", "Column roles")
    df = role_summary.copy()
    df["n_columns"] = pd.to_numeric(df["n_columns"], errors="coerce").fillna(0)
    df = df.sort_values("n_columns", ascending=True)
    palette = {"Feature": TEAL, "Target": GOLD, "Identifier": NAVY, "QC": "#6B5DD3", "Covariate": "#4E7AA8", "Ignore": MUTED}
    colors = [palette.get(str(r), MUTED) for r in df["role"]]
    fig, ax = plt.subplots(figsize=(9.5, max(4.5, 0.44 * len(df))))
    ax.barh(df["role"].astype(str), df["n_columns"].astype(float), color=colors)
    ax.set_xlabel("Number of columns", color=MUTED)
    _style(ax, "Column roles after mapping")
    for i, v in enumerate(df["n_columns"].astype(float)):
        ax.text(v + max(0.2, df["n_columns"].max() * 0.01), i, str(int(v)), va="center", fontsize=9, color=MUTED)
    return _save(fig, path)


def plot_group_counts(group_counts: pd.DataFrame, path: Path, group_variable: str = "task") -> Path:
    if group_counts is None or group_counts.empty or not {"group_variable", "level", "n_rows"}.issubset(group_counts.columns):
        return _empty(path, "No recognized grouping columns were detected. Add metadata with task, diagnosis, severity, subject, session, or modality to enable group-count plots.", "Group counts")
    df = group_counts[group_counts["group_variable"].astype(str).eq(group_variable)].copy()
    if df.empty:
        preferred = ["task", "diagnosis", "severity_bin", "modality", "subject_id", "session_id"]
        available = list(group_counts["group_variable"].astype(str).unique())
        chosen = next((p for p in preferred if p in available), available[0])
        df = group_counts[group_counts["group_variable"].astype(str).eq(chosen)].copy()
        group_variable = chosen
    df["n_rows"] = pd.to_numeric(df["n_rows"], errors="coerce").fillna(0)
    df = df.sort_values("n_rows", ascending=True).tail(30)
    fig, ax = plt.subplots(figsize=(10, max(4.8, 0.36 * len(df))))
    ax.barh(df["level"].astype(str), df["n_rows"].astype(float), color=TEAL)
    ax.set_xlabel("Rows / recordings", color=MUTED)
    _style(ax, f"Rows by {group_variable}")
    return _save(fig, path)


def plot_feature_family_counts(family_overview: pd.DataFrame, path: Path) -> Path:
    if family_overview is None or family_overview.empty or "family_or_subsystem" not in family_overview.columns or "n_features" not in family_overview.columns:
        return _empty(path, "No feature registry or subsystem labels were supplied. The plot will become informative after a feature registry / computation policy file is loaded.", "Feature families")
    df = family_overview.copy()
    df["n_features"] = pd.to_numeric(df["n_features"], errors="coerce").fillna(0)
    df = df.sort_values("n_features", ascending=True)
    fig, ax = plt.subplots(figsize=(10, max(4.8, 0.38 * len(df))))
    ax.barh(df["family_or_subsystem"].astype(str), df["n_features"].astype(float), color=TEAL)
    ax.set_xlabel("Features", color=MUTED)
    _style(ax, "Feature count by family / subsystem")
    return _save(fig, path)


def plot_row_missingness_distribution(row_missing: pd.DataFrame, path: Path) -> Path:
    if row_missing is None or row_missing.empty or "missing_fraction_feature_columns" not in row_missing.columns:
        return _empty(path, "No row-level missingness data were available.", "Row-level missingness")
    x = pd.to_numeric(row_missing["missing_fraction_feature_columns"], errors="coerce").dropna()
    if x.empty:
        return _empty(path, "No feature-row missingness values were available.", "Row-level missingness")
    fig, ax = plt.subplots(figsize=(9.8, 5.5))
    bins = np.linspace(0, 1, 21)
    ax.hist(x, bins=bins, color=TEAL, alpha=0.88, edgecolor="white")
    ax.axvline(0.20, color=GOLD, linestyle="--", linewidth=1.3, label="20% monitor")
    ax.axvline(0.50, color=RED, linestyle="--", linewidth=1.3, label="50% review")
    ax.axvline(float(x.median()), color=NAVY, linestyle="-", linewidth=1.5, label="median")
    ax.set_xlabel("Fraction of selected features missing in a row", color=MUTED)
    ax.set_ylabel("Rows / recordings", color=MUTED)
    _style(ax, "Row-level feature missingness")
    ax.legend(frameon=False, fontsize=8)
    return _save(fig, path)


def plot_missingness_by_group(group_summary: pd.DataFrame, path: Path, preferred: str = "task") -> Path:
    required = {"group_variable", "level", "n_rows", "mean_feature_missing_fraction"}
    if group_summary is None or group_summary.empty or not required.issubset(group_summary.columns):
        return _empty(path, "No grouping variables were available. Add metadata columns such as task, diagnosis, severity_bin, sex/gender, session, subject, or device.", "Missingness by group")
    df = group_summary.copy()
    available = list(df["group_variable"].astype(str).unique())
    group = preferred if preferred in available else next((g for g in ["task", "diagnosis", "severity_bin", "sex_or_gender", "device", "session", "subject"] if g in available), available[0])
    df = df[df["group_variable"].astype(str).eq(group)].copy()
    df["mean_feature_missing_fraction"] = pd.to_numeric(df["mean_feature_missing_fraction"], errors="coerce")
    df["n_rows"] = pd.to_numeric(df["n_rows"], errors="coerce").fillna(0)
    df = df.dropna(subset=["mean_feature_missing_fraction"]).sort_values("mean_feature_missing_fraction", ascending=True).tail(30)
    if df.empty:
        return _empty(path, "The selected grouping variable did not contain usable missingness values.", "Missingness by group")
    fig, ax = plt.subplots(figsize=(10.5, max(4.8, 0.36 * len(df))))
    colors = [RED if v >= 0.50 else GOLD if v >= 0.20 else TEAL for v in df["mean_feature_missing_fraction"]]
    labels = [f"{lev}  (n={int(n)})" for lev, n in zip(df["level"].astype(str), df["n_rows"])]
    ax.barh(labels, df["mean_feature_missing_fraction"], color=colors)
    ax.set_xlabel("Mean feature missingness", color=MUTED)
    ax.set_xlim(0, max(0.05, min(1.0, float(df["mean_feature_missing_fraction"].max()) * 1.15)))
    ax.axvline(0.20, color=GOLD, linestyle="--", linewidth=1, alpha=0.8)
    ax.axvline(0.50, color=RED, linestyle="--", linewidth=1, alpha=0.8)
    _style(ax, f"Mean feature missingness by {group}")
    return _save(fig, path)


def plot_missingness_family_summary(family_summary: pd.DataFrame, path: Path) -> Path:
    if family_summary is None or family_summary.empty or "family_or_subsystem" not in family_summary.columns:
        return _empty(path, "No feature-family/subsystem information was available. Load a feature registry/policy file to enable family-level missingness review.", "Missingness by family")
    df = family_summary.copy()
    df["mean_missing_fraction"] = pd.to_numeric(df["mean_missing_fraction"], errors="coerce")
    df = df.dropna(subset=["mean_missing_fraction"]).sort_values("mean_missing_fraction", ascending=True)
    if df.empty:
        return _empty(path, "No family-level missingness values were available.", "Missingness by family")
    fig, ax = plt.subplots(figsize=(10, max(4.8, 0.42 * len(df))))
    colors = [RED if v >= 0.50 else GOLD if v >= 0.20 else TEAL for v in df["mean_missing_fraction"]]
    ax.barh(df["family_or_subsystem"].astype(str), df["mean_missing_fraction"], color=colors)
    ax.set_xlabel("Mean feature missingness", color=MUTED)
    ax.set_xlim(0, max(0.05, min(1.0, float(df["mean_missing_fraction"].max()) * 1.15)))
    _style(ax, "Feature missingness by family / subsystem")
    return _save(fig, path)


def plot_comissing_heatmap(feature_df: pd.DataFrame, feature_cols: Sequence[str], path: Path, max_features: int | None = None) -> Path:
    cols = [c for c in list(feature_cols) if c in feature_df.columns]
    if not cols:
        return _empty(path, "No feature columns were available for co-missingness analysis.", "Co-missingness")
    miss_fr = feature_df[cols].isna().mean().sort_values(ascending=False)
    cols = miss_fr.index.tolist() if max_features is None else miss_fr.head(max_features).index.tolist()
    if len(cols) < 2:
        return _empty(path, "At least two feature columns are required for co-missingness analysis.", "Co-missingness")
    miss = feature_df[cols].isna().astype(float)
    mat = miss.T.dot(miss) / max(1, len(miss))
    fig, ax = plt.subplots(figsize=(max(8, 0.24 * len(cols)), max(7, 0.24 * len(cols))))
    im = ax.imshow(mat.to_numpy(), vmin=0, vmax=min(1, max(0.01, float(np.nanmax(mat.to_numpy())))), cmap="magma", aspect="auto")
    ax.set_title("Pairwise co-missingness among feature columns", fontsize=14, fontweight="bold", color=NAVY, pad=12)
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=90, fontsize=6, color=MUTED)
    ax.set_yticks(range(len(cols))); ax.set_yticklabels(cols, fontsize=6, color=MUTED)
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("co-missing fraction", color=MUTED)
    cbar.ax.tick_params(labelsize=8, colors=MUTED)
    return _save(fig, path)



def plot_distribution_review_summary(review: pd.DataFrame, path: Path) -> Path:
    if review is None or review.empty or "distribution_status" not in review.columns:
        return _empty(path, "No distribution review summary was available.", "Distribution review status")
    counts = review["distribution_status"].astype(str).value_counts().reindex(["ok", "monitor", "review"]).fillna(0)
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    colors = [TEAL, GOLD, RED]
    ax.bar(counts.index, counts.values, color=colors)
    ax.set_ylabel("Number of features", color=MUTED)
    _style(ax, "Distribution / outlier review status")
    for i, v in enumerate(counts.values):
        ax.text(i, v + max(0.1, counts.max() * 0.02), str(int(v)), ha="center", fontsize=10, color=MUTED)
    return _save(fig, path)


def plot_expected_range_flags(expected: pd.DataFrame, path: Path) -> Path:
    if expected is None or expected.empty or "fraction_outside_expected" not in expected.columns:
        return _empty(path, "No expected-range information was available. Load a registry/policy table with expected_low and expected_high to enable range plots.", "Expected-range flags")
    df = expected.copy()
    df["fraction_outside_expected"] = pd.to_numeric(df["fraction_outside_expected"], errors="coerce")
    df = df.dropna(subset=["fraction_outside_expected"]).sort_values("fraction_outside_expected", ascending=True).tail(35)
    if df.empty:
        return _empty(path, "Registry ranges were not available for the selected features.", "Expected-range flags")
    colors = [RED if v >= 0.20 else GOLD if v > 0 else TEAL for v in df["fraction_outside_expected"]]
    fig, ax = plt.subplots(figsize=(10.5, max(4.8, 0.34 * len(df))))
    ax.barh(df["feature"].astype(str), df["fraction_outside_expected"], color=colors)
    ax.set_xlabel("Fraction outside supplied expected range", color=MUTED)
    ax.set_xlim(0, max(0.05, min(1.0, float(df["fraction_outside_expected"].max()) * 1.15)))
    _style(ax, "Expected-range review by feature")
    return _save(fig, path)


def plot_selected_feature_distribution(
    df: pd.DataFrame,
    feature: str,
    path: Path,
    expected_low: float | None = None,
    expected_high: float | None = None,
    group_col: str | None = None,
) -> Path:
    if df is None or df.empty or feature not in df.columns:
        return _empty(path, "Selected feature was not found in the table.", "Selected feature distribution")
    x = pd.to_numeric(df[feature], errors="coerce")
    valid = x.dropna()
    if valid.empty:
        return _empty(path, "Selected feature has no valid numeric values.", f"Distribution: {feature}")
    fig, ax = plt.subplots(figsize=(10.5, 6.0))
    bins = min(30, max(6, int(np.sqrt(len(valid)))))
    ax.hist(valid, bins=bins, color=TEAL, alpha=0.85, edgecolor="white")
    med = float(valid.median())
    q1, q3 = np.nanpercentile(valid, [25, 75])
    ax.axvline(med, color=NAVY, linestyle="-", linewidth=1.8, label="median")
    ax.axvline(q1, color=MUTED, linestyle="--", linewidth=1.0, label="IQR")
    ax.axvline(q3, color=MUTED, linestyle="--", linewidth=1.0)
    if expected_low is not None and not np.isnan(expected_low):
        ax.axvline(expected_low, color=RED, linestyle=":", linewidth=1.5, label="expected range")
    if expected_high is not None and not np.isnan(expected_high):
        ax.axvline(expected_high, color=RED, linestyle=":", linewidth=1.5)
    ax.set_xlabel(feature, color=MUTED)
    ax.set_ylabel("Rows / recordings", color=MUTED)
    _style(ax, f"Distribution: {feature}")
    ax.legend(frameon=False, fontsize=8)
    return _save(fig, path)


def plot_group_feature_boxplot(df: pd.DataFrame, feature: str, path: Path, group_col: str | None = None) -> Path:
    if df is None or df.empty or feature not in df.columns:
        return _empty(path, "Selected feature was not found in the table.", "Feature by group")
    if group_col is None or group_col not in df.columns:
        candidates = [c for c in ["task", "diagnosis", "severity_bin", "modality", "sex", "gender"] if c in df.columns]
        group_col = candidates[0] if candidates else None
    if group_col is None:
        return _empty(path, "No grouping column was detected. Add task, diagnosis, severity_bin, modality, sex/gender, or metadata to enable grouped distribution plots.", "Feature by group")
    work = df[[feature, group_col]].copy()
    work[feature] = pd.to_numeric(work[feature], errors="coerce")
    work = work.dropna(subset=[feature, group_col])
    if work.empty or work[group_col].nunique() < 2:
        return _empty(path, "The selected grouping column does not have at least two usable groups.", "Feature by group")
    counts = work[group_col].astype(str).value_counts().head(12)
    groups = list(counts.index)
    data = [work.loc[work[group_col].astype(str).eq(g), feature].to_numpy(dtype=float) for g in groups]
    fig, ax = plt.subplots(figsize=(max(9, 0.8 * len(groups)), 6.2))
    ax.boxplot(data, tick_labels=[f"{g}\n(n={len(d)})" for g, d in zip(groups, data)], showfliers=False)
    for i, d in enumerate(data, start=1):
        if len(d):
            jitter = np.linspace(-0.14, 0.14, len(d)) if len(d) > 1 else np.array([0.0])
            ax.scatter(np.full(len(d), i) + jitter, d, s=18, alpha=0.65, color=TEAL)
    ax.set_ylabel(feature, color=MUTED)
    _style(ax, f"{feature} by {group_col}")
    ax.tick_params(axis="x", labelrotation=35)
    return _save(fig, path)



def _plot_context_col(df: pd.DataFrame, aliases: Sequence[str]) -> str | None:
    if df is None or df.empty:
        return None
    norm = {str(c).lower().replace(" ", "_").replace("-", "_"): c for c in df.columns}
    for a in aliases:
        key = str(a).lower().replace(" ", "_").replace("-", "_")
        if key in norm:
            return norm[key]
    return None


def _clean_context_series(s: pd.Series) -> pd.Series:
    out = s.astype("string").fillna("").str.strip()
    return out.loc[out.ne("") & ~out.str.lower().isin(["nan", "none", "<na>", "nat"])]


def plot_overview_dataset_structure(df: pd.DataFrame, path: Path) -> Path:
    """Dataset-structure overview: subjects, sex/gender, diagnosis, tasks, sessions, repeats."""
    if df is None or df.empty:
        return _empty(path, "No analysis table was available.", "Dataset structure")
    subject_col = _plot_context_col(df, ["subject_id", "participant_id", "patient_id", "metadata__subject_id"])
    task_col = _plot_context_col(df, ["task", "task_name", "prompt", "metadata__task"])
    sex_col = _plot_context_col(df, ["sex_or_gender", "sex", "gender", "metadata__sex_or_gender", "metadata__sex", "metadata__gender"])
    diagnosis_col = _plot_context_col(df, ["diagnosis", "dx", "disease_group", "group_label", "metadata__diagnosis"])
    session_col = _plot_context_col(df, ["session_id", "visit_id", "timepoint", "recording_date", "assessment_date", "metadata__recording_date"])
    iteration_col = _plot_context_col(df, ["iteration", "metadata__iteration"])

    subject_values = _clean_context_series(df[subject_col]) if subject_col else pd.Series(dtype=str)
    task_values = _clean_context_series(df[task_col]) if task_col else pd.Series(dtype=str)
    sex_counts = _clean_context_series(df[sex_col]).value_counts().head(10) if sex_col else pd.Series(dtype=int)
    dx_counts = _clean_context_series(df[diagnosis_col]).value_counts().head(10) if diagnosis_col else pd.Series(dtype=int)
    task_counts = task_values.value_counts().head(14) if task_col else pd.Series(dtype=int)

    n_subjects = int(subject_values.nunique()) if subject_col else 0
    n_tasks = int(task_values.nunique()) if task_col else 0
    repeated_subjects = 0
    sessions_per_subject = pd.Series(dtype=float)
    if subject_col:
        if session_col:
            tmp = df[[subject_col, session_col]].copy()
            tmp[subject_col] = tmp[subject_col].astype("string").fillna("").str.strip()
            tmp[session_col] = tmp[session_col].astype("string").fillna("").str.strip()
            tmp = tmp.loc[tmp[subject_col].ne("") & tmp[session_col].ne("")]
            if not tmp.empty:
                sessions_per_subject = tmp.groupby(subject_col)[session_col].nunique().sort_values(ascending=False)
        if sessions_per_subject.empty:
            sessions_per_subject = df[subject_col].astype("string").fillna("").str.strip().value_counts()
        if task_col and iteration_col:
            tmp = df[[subject_col, task_col, iteration_col]].copy()
            for c in [subject_col, task_col, iteration_col]:
                tmp[c] = tmp[c].astype("string").fillna("").str.strip()
            tmp = tmp.loc[tmp[subject_col].ne("") & tmp[task_col].ne("") & tmp[iteration_col].ne("")]
            if not tmp.empty:
                rep = tmp.groupby([subject_col, task_col])[iteration_col].nunique().reset_index(name="n_iterations")
                repeated_subjects = int(rep.loc[rep["n_iterations"].ge(2), subject_col].nunique())
        elif task_col:
            tmp = df[[subject_col, task_col]].copy()
            tmp[subject_col] = tmp[subject_col].astype("string").fillna("").str.strip()
            tmp[task_col] = tmp[task_col].astype("string").fillna("").str.strip()
            rep = tmp.groupby([subject_col, task_col]).size().reset_index(name="n_records")
            repeated_subjects = int(rep.loc[rep["n_records"].ge(2), subject_col].nunique())

    max_labels = int(c.groupby("section").size().max()) if not c.empty else 1
    fig_h = max(10.2, 8.2 + 0.18 * min(max_labels, 24))
    fig = plt.figure(figsize=(15.8, fig_h))
    gs = fig.add_gridspec(3, 2, height_ratios=[0.9, 1.35, 1.55], hspace=0.58, wspace=0.36)
    ax0 = fig.add_subplot(gs[0, :]); ax0.axis("off")
    ax0.text(0.00, 0.82, "Dataset design context", fontsize=18, fontweight="bold", color=NAVY, va="center")
    ax0.text(0.00, 0.50, "Counts from the active analysis table after feature, metadata, QC, and filename context mapping.", fontsize=10.5, color=MUTED, va="center")
    cards = [("Subjects", n_subjects), ("Tasks", n_tasks), ("Rows/files", len(df)), ("Repeated-analysis subjects", repeated_subjects)]
    for i, (lab, val) in enumerate(cards):
        x = 0.02 + i * 0.24
        ax0.add_patch(plt.Rectangle((x, 0.04), 0.20, 0.30, transform=ax0.transAxes, color="#F7FAFD", ec=GRID, lw=0.8))
        ax0.text(x + 0.015, 0.25, lab, transform=ax0.transAxes, fontsize=9, color=MUTED, fontweight="bold", va="center")
        ax0.text(x + 0.015, 0.11, str(val), transform=ax0.transAxes, fontsize=18, color=NAVY, fontweight="bold", va="center")

    axes = [fig.add_subplot(gs[1,0]), fig.add_subplot(gs[1,1]), fig.add_subplot(gs[2,0]), fig.add_subplot(gs[2,1])]
    plot_items = [
        (axes[0], sex_counts, "Sex / gender counts", "subjects/rows"),
        (axes[1], dx_counts, "Diagnosis counts", "subjects/rows"),
        (axes[2], task_counts, "Task counts", "records"),
        (axes[3], sessions_per_subject.value_counts().sort_index().head(15), "Sessions per subject", "subjects"),
    ]
    for ax, counts, title, xlabel in plot_items:
        if counts is None or len(counts) == 0:
            ax.text(0.5, 0.5, "not available", ha="center", va="center", color=MUTED)
            ax.set_title(title, fontsize=12, fontweight="bold", color=NAVY)
            ax.axis("off")
            continue
        labels = [str(x)[:36] for x in counts.index]
        vals = counts.astype(float).to_numpy()
        ax.barh(labels[::-1], vals[::-1], color=TEAL)
        for j, v in enumerate(vals[::-1]):
            ax.text(v + max(vals) * 0.015, j, f"{int(v)}", va="center", fontsize=8, color=MUTED)
        ax.set_xlabel(xlabel, color=MUTED)
        ax.tick_params(axis="y", labelsize=8.2)
        _style(ax, title)
    fig.subplots_adjust(left=0.17, right=0.98)
    return _save(fig, path)


def plot_role_mapping_summary(role_summary: pd.DataFrame, path: Path) -> Path:
    if role_summary is None or role_summary.empty:
        return _empty(path, "No role mapping summary was available.", "Role mapping summary")
    df = role_summary.copy()
    role_col = "role" if "role" in df.columns else df.columns[0]
    count_col = "n_columns" if "n_columns" in df.columns else ("count" if "count" in df.columns else None)
    if count_col is None:
        df["n_columns"] = 1
        count_col = "n_columns"
    df[count_col] = pd.to_numeric(df[count_col], errors="coerce").fillna(0)
    df = df.loc[df[count_col].gt(0)].sort_values(count_col, ascending=True).tail(24)
    if df.empty:
        return _empty(path, "No mapped columns were counted.", "Role mapping summary")
    fig, ax = plt.subplots(figsize=(11.5, max(5.2, 0.38 * len(df))))
    ax.barh(df[role_col].astype(str), df[count_col], color=TEAL)
    ax.set_xlabel("Number of columns", color=MUTED)
    _style(ax, "Accepted role mapping summary")
    for i, v in enumerate(df[count_col]):
        ax.text(float(v) + max(float(df[count_col].max()) * 0.015, 0.2), i, f"{int(v)}", va="center", fontsize=8.5, color=MUTED)
    return _save(fig, path)


def plot_subject_task_coverage_bars(df: pd.DataFrame, path: Path) -> Path:
    if df is None or df.empty:
        return _empty(path, "No analysis table was available.", "Subject × task coverage")
    subject_col = _plot_context_col(df, ["subject_id", "participant_id", "patient_id", "metadata__subject_id"])
    task_col = _plot_context_col(df, ["task", "task_name", "prompt", "metadata__task"])
    iteration_col = _plot_context_col(df, ["iteration", "metadata__iteration"])
    if subject_col is None or task_col is None:
        return _empty(path, "Subject and task columns are required for subject × task coverage.", "Subject × task coverage")
    work = df[[subject_col, task_col] + ([iteration_col] if iteration_col else [])].copy()
    for c in work.columns:
        work[c] = work[c].astype("string").fillna("").str.strip()
    work = work.loc[work[subject_col].ne("") & work[task_col].ne("")]
    if work.empty:
        return _empty(path, "No non-empty subject/task records were available.", "Subject × task coverage")
    records = work[task_col].value_counts()
    subjects = work.groupby(task_col)[subject_col].nunique()
    if iteration_col:
        rep = work.loc[work[iteration_col].ne("")].groupby([subject_col, task_col])[iteration_col].nunique().reset_index(name="n_iterations")
        repeated = rep.loc[rep["n_iterations"].ge(2)].groupby(task_col)[subject_col].nunique()
    else:
        rep = work.groupby([subject_col, task_col]).size().reset_index(name="n_records")
        repeated = rep.loc[rep["n_records"].ge(2)].groupby(task_col)[subject_col].nunique()
    summary = pd.DataFrame({"records": records, "subjects": subjects, "repeated_subjects": repeated}).fillna(0)
    summary = summary.sort_values(["subjects", "records"], ascending=True).tail(24)
    fig, ax = plt.subplots(figsize=(12.4, max(5.5, 0.42 * len(summary))))
    y = np.arange(len(summary))
    width = 0.25
    ax.barh(y - width, summary["records"], height=width, color=TEAL, label="records")
    ax.barh(y, summary["subjects"], height=width, color=GOLD, label="subjects")
    ax.barh(y + width, summary["repeated_subjects"], height=width, color=NAVY, label="subjects with ≥2 iterations/records")
    ax.set_yticks(y); ax.set_yticklabels([str(x)[:42] for x in summary.index], color=MUTED, fontsize=8.5)
    ax.set_xlabel("Count", color=MUTED)
    ax.legend(frameon=False, fontsize=8)
    _style(ax, "Subject × task coverage counts")
    fig.subplots_adjust(left=0.26, right=0.98)
    return _save(fig, path)


def plot_overview_readiness_scorecard(readiness: pd.DataFrame, path: Path) -> Path:
    if readiness is None or readiness.empty or not {"dimension", "score_0_100"}.issubset(readiness.columns):
        return _empty(path, "No readiness summary was available.", "Overview readiness")
    df = readiness.copy()
    df["score_0_100"] = pd.to_numeric(df["score_0_100"], errors="coerce").fillna(0)
    df = df.sort_values("score_0_100", ascending=True)
    colors = [TEAL if v >= 80 else GOLD if v >= 50 else RED if v > 0 else MUTED for v in df["score_0_100"]]
    fig, ax = plt.subplots(figsize=(10.5, max(4.8, 0.54 * len(df))))
    ax.barh(df["dimension"].astype(str), df["score_0_100"], color=colors)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Orientation score (0–100)", color=MUTED)
    ax.axvline(50, color=GOLD, linestyle="--", linewidth=1, alpha=0.8)
    ax.axvline(80, color=TEAL, linestyle="--", linewidth=1, alpha=0.8)
    _style(ax, "Dataset readiness orientation")
    for i, v in enumerate(df["score_0_100"]):
        ax.text(min(99, float(v) + 1.5), i, f"{float(v):.0f}", va="center", fontsize=9, color=MUTED)
    return _save(fig, path)


def plot_dataset_design_tiles(design: pd.DataFrame, path: Path) -> Path:
    """Professional dataset-context status board for Overview."""
    required = {"variable_type", "column", "n_unique", "n_missing"}
    if design is None or design.empty or not required.issubset(design.columns):
        return _empty(path, "No dataset-design summary was available.", "Dataset design")

    df = design.copy().head(12)
    for col in ["variable_type", "column", "status", "top_values"]:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].astype("string").fillna("").astype(str)
    df["n_unique"] = pd.to_numeric(df["n_unique"], errors="coerce").fillna(0).astype(int)
    df["n_missing"] = pd.to_numeric(df["n_missing"], errors="coerce").fillna(0).astype(int)
    if "status" not in df.columns or (df["status"].astype(str).str.len().sum() == 0):
        df["status"] = np.where(df["column"].eq("not_detected"), "missing", np.where(df["n_unique"].gt(0), "detected", "empty_column"))

    status_label = {"detected": "Detected", "empty_column": "Empty", "missing": "Missing"}
    status_color = {"detected": TEAL, "empty_column": GOLD, "missing": RED}
    n = len(df)

    fig, ax = plt.subplots(figsize=(13.8, max(5.4, 0.58 * n + 2.4)))
    ax.set_xlim(0, 100)
    ax.set_ylim(-1.0, n + 1.15)
    ax.axis("off")

    ax.text(0, n + 0.72, "Dataset design context", fontsize=17, fontweight="bold", color=NAVY, va="center")
    ax.text(0, n + 0.20, "Detected recording, task, and clinical context used to interpret downstream feature views.", fontsize=10.2, color=MUTED, va="center")

    headers = [(0, "Context"), (24, "Detected column"), (54, "Status"), (70, "Observed values")]
    for x, h in headers:
        ax.text(x, n - 0.18, h, fontsize=9.5, fontweight="bold", color=NAVY, va="bottom")

    for i, (_, r) in enumerate(df.iterrows()):
        yy = n - 1 - i
        bg = "#F7FAFD" if i % 2 == 0 else "#FFFFFF"
        ax.add_patch(plt.Rectangle((0, yy - 0.42), 100, 0.82, color=bg, ec="#E4ECF4", lw=0.35, zorder=0))

        stat = str(r.get("status", "")) or ("detected" if str(r.get("column", "")) != "not_detected" and int(r.get("n_unique", 0)) > 0 else "missing")
        color = status_color.get(stat, MUTED)
        label = status_label.get(stat, stat.replace("_", " ").title())

        variable = str(r.get("variable_type", "variable")).replace("_", " ").title()
        column = str(r.get("column", "not_detected"))
        if len(column) > 34:
            column = column[:31] + "..."

        values = f"{int(r.get('n_unique', 0))} unique · {int(r.get('n_missing', 0))} missing"
        top_values = str(r.get("top_values", ""))
        if top_values and top_values.lower() not in {"nan", "none", "<na>"}:
            top_values = top_values[:70] + ("..." if len(top_values) > 70 else "")
            values = values + " · " + top_values

        ax.text(0, yy, variable, fontsize=10.2, color=NAVY, va="center", fontweight="bold")
        ax.text(24, yy, column, fontsize=9.3, color=TEAL if stat == "detected" else MUTED, va="center")

        # pill-style status marker
        ax.add_patch(plt.Rectangle((54, yy - 0.22), 11.5, 0.44, color=color, alpha=0.14, ec=color, lw=0.8, zorder=1))
        ax.text(59.75, yy, label, fontsize=8.6, color=color, va="center", ha="center", fontweight="bold")

        ax.text(70, yy, values, fontsize=8.8, color=MUTED, va="center")

    detected = int((df["status"].astype(str) == "detected").sum())
    ax.text(0, -0.62, f"{detected}/{n} context fields detected. Missing clinical fields are acceptable when metadata is absent, but should be reviewed before disease/severity interpretation.",
            fontsize=9.4, color=MUTED, va="center")
    return _save(fig, path)

def plot_feature_quality_landscape(quality: pd.DataFrame, path: Path) -> Path:
    required = {"feature", "missing_fraction", "robust_outlier_fraction", "quality_status"}
    if quality is None or quality.empty or not required.issubset(quality.columns):
        return _empty(path, "No feature-quality landscape was available.", "Feature quality landscape")
    df = quality.copy()
    df["missing_fraction"] = pd.to_numeric(df["missing_fraction"], errors="coerce")
    df["robust_outlier_fraction"] = pd.to_numeric(df["robust_outlier_fraction"], errors="coerce")
    df["n_valid"] = pd.to_numeric(df.get("n_valid", 1), errors="coerce").fillna(1)
    df = df.dropna(subset=["missing_fraction", "robust_outlier_fraction"])
    if df.empty:
        return _empty(path, "No numeric feature-quality values were available.", "Feature quality landscape")
    colors = df["quality_status"].map({"ok": TEAL, "monitor": GOLD, "review": RED}).fillna(MUTED).tolist()
    sizes = np.clip(20 + 2.5 * np.sqrt(df["n_valid"].to_numpy(dtype=float)), 22, 120)
    fig, ax = plt.subplots(figsize=(10.8, 6.4))
    ax.scatter(df["missing_fraction"], df["robust_outlier_fraction"], s=sizes, c=colors, alpha=0.78, edgecolors="white", linewidth=0.6)
    ax.axvline(0.20, color=GOLD, linestyle="--", linewidth=1.0, alpha=0.8)
    ax.axvline(0.50, color=RED, linestyle="--", linewidth=1.0, alpha=0.8)
    ax.axhline(0.05, color=GOLD, linestyle="--", linewidth=1.0, alpha=0.8)
    ax.axhline(0.20, color=RED, linestyle="--", linewidth=1.0, alpha=0.8)
    ax.set_xlim(-0.02, min(1.0, max(0.05, float(df["missing_fraction"].max()) * 1.12)))
    ax.set_ylim(-0.01, min(1.0, max(0.05, float(df["robust_outlier_fraction"].max()) * 1.20)))
    ax.set_xlabel("Missing fraction", color=MUTED)
    ax.set_ylabel("Robust outlier fraction", color=MUTED)
    _style(ax, "Feature quality landscape")
    label_df = df.sort_values(["quality_status", "missing_fraction", "robust_outlier_fraction"], ascending=[False, False, False]).head(10)
    for _, r in label_df.iterrows():
        if str(r.get("quality_status")) in {"monitor", "review"}:
            ax.text(float(r["missing_fraction"]) + 0.005, float(r["robust_outlier_fraction"]) + 0.003, str(r["feature"])[:24], fontsize=7, color=MUTED)
    return _save(fig, path)


def plot_feature_family_quality(family_overview: pd.DataFrame, path: Path) -> Path:
    if family_overview is None or family_overview.empty or "family_or_subsystem" not in family_overview.columns:
        return _empty(path, "No feature-family metadata was available. Load a registry/policy table to make this plot more informative.", "Family quality overview")
    df = family_overview.copy()
    for col in ["n_features", "mean_missing_fraction", "n_numeric_features"]:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    df = df.sort_values("n_features", ascending=True)
    fig, ax = plt.subplots(figsize=(10.5, max(4.8, 0.44 * len(df))))
    colors = [RED if m >= 0.50 else GOLD if m >= 0.20 else TEAL for m in df["mean_missing_fraction"]]
    labels = [f"{fam}  (miss={m:.1%})" for fam, m in zip(df["family_or_subsystem"].astype(str), df["mean_missing_fraction"])]
    ax.barh(labels, df["n_features"], color=colors)
    ax.set_xlabel("Number of mapped features", color=MUTED)
    _style(ax, "Feature-family coverage and missingness")
    return _save(fig, path)


def plot_subject_task_matrix(df: pd.DataFrame, path: Path) -> Path:
    """Backward-compatible Overview subject/task coverage, now rendered as count bars rather than a heatmap."""
    return plot_subject_task_coverage_bars(df, path)


def plot_distribution_shape_summary(shape: pd.DataFrame, path: Path) -> Path:
    if shape is None or shape.empty or "priority" not in shape.columns:
        return _empty(path, "No distribution-shape audit was available.", "Distribution shape audit")
    order = ["ok", "monitor", "review"]
    counts = shape["priority"].astype(str).value_counts().reindex(order).fillna(0)
    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    colors = [TEAL, GOLD, RED]
    ax.bar(counts.index, counts.values, color=colors)
    ax.set_ylabel("Number of features", color=MUTED)
    _style(ax, "Distribution-shape review priority")
    for i, v in enumerate(counts.values):
        ax.text(i, v + max(0.1, counts.max() * 0.03), str(int(v)), ha="center", fontsize=10, color=MUTED)
    ax.text(0.01, -0.20, "Review priority integrates spread, skew/tails, floor/ceiling effects, missingness, robust outliers, and expected-range flags.",
            transform=ax.transAxes, fontsize=9, color=MUTED, va="top")
    return _save(fig, path)


def plot_distribution_shape_landscape(shape: pd.DataFrame, path: Path) -> Path:
    required = {"feature", "skew_proxy", "tail_ratio", "priority"}
    if shape is None or shape.empty or not required.issubset(shape.columns):
        return _empty(path, "No shape landscape data were available.", "Distribution shape landscape")
    df = shape.copy()
    df["skew_proxy"] = pd.to_numeric(df["skew_proxy"], errors="coerce")
    df["tail_ratio"] = pd.to_numeric(df["tail_ratio"], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["skew_proxy", "tail_ratio"])
    if df.empty:
        return _empty(path, "No usable skew/tail values were available.", "Distribution shape landscape")
    color_map = {"ok": TEAL, "monitor": GOLD, "review": RED}
    colors = [color_map.get(str(p), MUTED) for p in df["priority"]]
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    ax.scatter(df["skew_proxy"], df["tail_ratio"], s=52, c=colors, alpha=0.78, edgecolors="white", linewidths=0.7)
    ax.axvline(0, color=GRID, linewidth=1)
    ax.axvline(0.75, color=GOLD, linestyle="--", linewidth=1)
    ax.axvline(-0.75, color=GOLD, linestyle="--", linewidth=1)
    ax.axhline(4.5, color=GOLD, linestyle="--", linewidth=1)
    ax.set_xlabel("Skew proxy: (mean − median) / SD", color=MUTED)
    ax.set_ylabel("Tail ratio: (q95 − q05) / IQR", color=MUTED)
    _style(ax, "Distribution shape landscape")
    # label highest-risk points only
    lab = df[df["priority"].astype(str).eq("review")].head(10)
    for _, r in lab.iterrows():
        ax.text(float(r["skew_proxy"]), float(r["tail_ratio"]), "  " + str(r["feature"])[:28], fontsize=7, color=NAVY, alpha=0.85)
    return _save(fig, path)


def plot_row_outlier_burden(row_burden: pd.DataFrame, path: Path) -> Path:
    if row_burden is None or row_burden.empty or "n_flagged_features" not in row_burden.columns:
        return _empty(path, "No row-level outlier burden was available.", "Row-level outlier burden")
    df = row_burden.copy()
    df["n_flagged_features"] = pd.to_numeric(df["n_flagged_features"], errors="coerce").fillna(0)
    df = df.sort_values("n_flagged_features", ascending=False).head(40).sort_values("n_flagged_features", ascending=True)
    label_col = next((c for c in ["file_name", "record_key", "subject_id", "row_index"] if c in df.columns), "row_index")
    colors = [RED if str(v) == "review" else GOLD if str(v) == "monitor" else TEAL for v in df.get("review_level", pd.Series(["ok"] * len(df)))]
    fig, ax = plt.subplots(figsize=(11, max(4.8, 0.32 * len(df))))
    ax.barh(df[label_col].astype(str), df["n_flagged_features"], color=colors)
    ax.set_xlabel("Number of flagged features on row / recording", color=MUTED)
    _style(ax, "Rows accumulating multiple feature flags")
    return _save(fig, path)


def plot_variance_screen(dist: pd.DataFrame, path: Path) -> Path:
    if dist is None or dist.empty or "feature" not in dist.columns:
        return _empty(path, "No distribution summary was available.", "Variance screen")
    df = dist.copy()
    df["iqr"] = pd.to_numeric(df.get("iqr"), errors="coerce")
    df["unique_values"] = pd.to_numeric(df.get("unique_values"), errors="coerce")
    df["zero_variance"] = df.get("zero_variance", False).fillna(False).astype(bool)
    df["near_zero_variance"] = df.get("near_zero_variance", False).fillna(False).astype(bool)
    df = df.sort_values(["zero_variance", "near_zero_variance", "iqr"], ascending=[False, False, True]).head(35).iloc[::-1]
    colors = [RED if z else GOLD if nz else TEAL for z, nz in zip(df["zero_variance"], df["near_zero_variance"])]
    fig, ax = plt.subplots(figsize=(10.5, max(4.8, 0.34 * len(df))))
    ax.barh(df["feature"].astype(str), df["iqr"].fillna(0), color=colors)
    ax.set_xlabel("Interquartile range (native units)", color=MUTED)
    _style(ax, "Zero / near-zero variance screen")
    return _save(fig, path)




def plot_distribution_shape_story(shape: pd.DataFrame, path: Path) -> Path:
    """Clinician-facing combined shape triage: priority counts plus skew/tail landscape."""
    required = {"feature", "priority"}
    if shape is None or shape.empty or not required.issubset(shape.columns):
        return _empty(path, "No distribution-shape audit was available for this task scope.", "Distribution shape audit")
    df = shape.copy()
    df["priority"] = df["priority"].astype(str).str.lower()
    order = ["ok", "monitor", "review"]
    counts = df["priority"].value_counts().reindex(order).fillna(0)
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.8), gridspec_kw={"width_ratios": [0.85, 1.45]})
    colors = [TEAL, GOLD, RED]
    axes[0].bar(counts.index, counts.values, color=colors)
    axes[0].set_ylabel("Number of features", color=MUTED)
    _style(axes[0], "Shape priority")
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + max(0.1, counts.max() * 0.04), str(int(v)), ha="center", fontsize=10, color=MUTED)

    ax = axes[1]
    if {"skew_proxy", "tail_ratio"}.issubset(df.columns):
        df["skew_proxy"] = pd.to_numeric(df["skew_proxy"], errors="coerce")
        df["tail_ratio"] = pd.to_numeric(df["tail_ratio"], errors="coerce")
        xy = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["skew_proxy", "tail_ratio"])
    else:
        xy = pd.DataFrame()
    if xy.empty:
        ax.text(0.5, 0.5, "No usable skew/tail values", ha="center", va="center", color=MUTED)
        ax.axis("off")
    else:
        cmap = {"ok": TEAL, "monitor": GOLD, "review": RED}
        ax.scatter(xy["skew_proxy"], xy["tail_ratio"], s=52, c=[cmap.get(str(v), MUTED) for v in xy["priority"]], alpha=0.78, edgecolors="white", linewidths=0.7)
        ax.axvline(0, color=GRID, linewidth=1)
        ax.axvline(0.75, color=GOLD, linestyle="--", linewidth=1)
        ax.axvline(-0.75, color=GOLD, linestyle="--", linewidth=1)
        ax.axhline(4.5, color=GOLD, linestyle="--", linewidth=1)
        ax.set_xlabel("Skew proxy: (mean − median) / SD", color=MUTED)
        ax.set_ylabel("Tail ratio: (q95 − q05) / IQR", color=MUTED)
        _style(ax, "Skew / tail landscape")
        lab = xy[xy["priority"].astype(str).eq("review")].head(8)
        for _, r in lab.iterrows():
            ax.text(float(r["skew_proxy"]), float(r["tail_ratio"]), "  " + str(r["feature"])[:24], fontsize=7, color=NAVY, alpha=0.85)
    fig.suptitle("Distribution shape and tail audit", fontsize=15, fontweight="bold", color=NAVY)
    return _save(fig, path)


def plot_distribution_outlier_range_story(expected: pd.DataFrame, outliers: pd.DataFrame, review: pd.DataFrame, path: Path) -> Path:
    """Combined feature-level plausibility panel: expected-range flags plus robust outlier evidence."""
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 6.2), gridspec_kw={"width_ratios": [1.1, 1.1]})
    # Expected range panel.
    ax = axes[0]
    if expected is None or expected.empty or "fraction_outside_expected" not in expected.columns:
        ax.text(0.5, 0.5, "No registry expected ranges available", ha="center", va="center", color=MUTED, wrap=True)
        ax.axis("off")
    else:
        er = expected.copy()
        er["fraction_outside_expected"] = pd.to_numeric(er["fraction_outside_expected"], errors="coerce")
        er = er.dropna(subset=["fraction_outside_expected"]).sort_values("fraction_outside_expected", ascending=False).head(18).iloc[::-1]
        if er.empty:
            ax.text(0.5, 0.5, "No outside-range features", ha="center", va="center", color=MUTED)
            ax.axis("off")
        else:
            colors = [RED if v >= 0.20 else GOLD if v > 0 else TEAL for v in er["fraction_outside_expected"]]
            ax.barh(er["feature"].astype(str), er["fraction_outside_expected"], color=colors)
            ax.set_xlabel("Fraction outside expected range", color=MUTED)
            ax.set_xlim(0, max(0.05, min(1.0, float(er["fraction_outside_expected"].max()) * 1.15)))
            _style(ax, "Expected-range flags")
    # Robust outlier count panel.
    ax = axes[1]
    if outliers is None or outliers.empty or "feature" not in outliers.columns:
        ax.text(0.5, 0.5, "No robust outlier flags", ha="center", va="center", color=MUTED)
        ax.axis("off")
    else:
        cnt = outliers["feature"].astype(str).value_counts().head(18).sort_values(ascending=True)
        ax.barh(cnt.index, cnt.values, color=GOLD)
        ax.set_xlabel("Robust outlier rows", color=MUTED)
        _style(ax, "Outlier counts")
    fig.suptitle("Range and outlier triage", fontsize=15, fontweight="bold", color=NAVY)
    return _save(fig, path)


def plot_row_outlier_story(row_burden: pd.DataFrame, path: Path) -> Path:
    """Readable recording-level burden panel: burden distribution and top review rows."""
    if row_burden is None or row_burden.empty or "n_flagged_features" not in row_burden.columns:
        return _empty(path, "No row-level outlier burden was available.", "Recording outlier burden")
    df = row_burden.copy()
    df["n_flagged_features"] = pd.to_numeric(df["n_flagged_features"], errors="coerce").fillna(0)
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 6.0), gridspec_kw={"width_ratios": [0.85, 1.35]})
    ax = axes[0]
    if "review_level" in df.columns:
        order = ["ok", "monitor", "review"]
        counts = df["review_level"].astype(str).value_counts().reindex(order).fillna(0)
        ax.bar(counts.index, counts.values, color=[TEAL, GOLD, RED])
        ax.set_ylabel("Recordings", color=MUTED)
        _style(ax, "Recording triage")
        for i, v in enumerate(counts.values):
            ax.text(i, v + max(0.1, counts.max() * 0.04), str(int(v)), ha="center", fontsize=10, color=MUTED)
    else:
        ax.hist(df["n_flagged_features"], bins=min(25, max(5, int(np.sqrt(len(df))))), color=TEAL, edgecolor="white")
        ax.set_xlabel("Flagged features per recording", color=MUTED)
        ax.set_ylabel("Recordings", color=MUTED)
        _style(ax, "Burden distribution")
    top = df.sort_values("n_flagged_features", ascending=False).head(25).iloc[::-1]
    ax = axes[1]
    label_col = next((c for c in ["file_name", "record_key", "subject_id", "row_index"] if c in top.columns), "row_index")
    colors = [RED if str(v) == "review" else GOLD if str(v) == "monitor" else TEAL for v in top.get("review_level", pd.Series(["ok"] * len(top)))]
    ax.barh(top[label_col].astype(str), top["n_flagged_features"], color=colors)
    ax.set_xlabel("Number of flagged features", color=MUTED)
    _style(ax, "Highest-burden recordings")
    fig.suptitle("Recording outlier burden", fontsize=15, fontweight="bold", color=NAVY)
    return _save(fig, path)


def plot_selected_feature_diagnostic(
    df: pd.DataFrame,
    feature: str,
    path: Path,
    expected_low: float | None = None,
    expected_high: float | None = None,
    group_col: str | None = None,
) -> Path:
    if df is None or df.empty or feature not in df.columns:
        return _empty(path, "Selected feature was not found in the table.", "Selected-feature diagnostic")
    x_all = pd.to_numeric(df[feature], errors="coerce")
    valid = x_all.dropna()
    if valid.empty:
        return _empty(path, "Selected feature has no valid numeric values.", f"Diagnostic: {feature}")
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.2))
    ax = axes[0, 0]
    bins = min(34, max(6, int(np.sqrt(len(valid)))))
    ax.hist(valid, bins=bins, color=TEAL, alpha=0.88, edgecolor="white")
    med = float(valid.median())
    q1, q3 = np.nanpercentile(valid, [25, 75])
    ax.axvline(med, color=NAVY, linewidth=1.8, label="median")
    ax.axvline(q1, color=MUTED, linestyle="--", linewidth=1.0, label="IQR")
    ax.axvline(q3, color=MUTED, linestyle="--", linewidth=1.0)
    if expected_low is not None and not np.isnan(expected_low):
        ax.axvline(expected_low, color=RED, linestyle=":", linewidth=1.5, label="expected range")
    if expected_high is not None and not np.isnan(expected_high):
        ax.axvline(expected_high, color=RED, linestyle=":", linewidth=1.5)
    ax.set_xlabel(feature, color=MUTED); ax.set_ylabel("Rows", color=MUTED)
    _style(ax, "Histogram with median, IQR, expected range")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[0, 1]
    ax.boxplot(valid.to_numpy(dtype=float), vert=False, showfliers=False)
    y = np.linspace(0.88, 1.12, len(valid)) if len(valid) > 1 else np.array([1.0])
    ax.scatter(valid, y, s=18, alpha=0.55, color=TEAL)
    ax.set_yticks([]); ax.set_xlabel(feature, color=MUTED)
    _style(ax, "Box/strip view: spread and extreme rows")

    ax = axes[1, 0]
    sorted_x = np.sort(valid.to_numpy(dtype=float))
    ecdf = np.arange(1, len(sorted_x) + 1) / len(sorted_x)
    ax.plot(sorted_x, ecdf, color=TEAL, linewidth=2)
    ax.axhline(0.5, color=GRID, linestyle="--", linewidth=1)
    ax.set_xlabel(feature, color=MUTED); ax.set_ylabel("Cumulative fraction", color=MUTED)
    _style(ax, "Empirical cumulative distribution")

    ax = axes[1, 1]
    n = len(sorted_x)
    if n >= 3 and float(np.nanstd(sorted_x)) > 0:
        from statistics import NormalDist
        probs = (np.arange(1, n + 1) - 0.5) / n
        theo = np.array([NormalDist().inv_cdf(float(p)) for p in probs])
        ax.scatter(theo, sorted_x, s=18, alpha=0.65, color=TEAL)
        # reference line through q1/q3
        tq1, tq3 = np.percentile(theo, [25, 75])
        xq1, xq3 = np.percentile(sorted_x, [25, 75])
        slope = (xq3 - xq1) / (tq3 - tq1) if tq3 != tq1 else 0
        intercept = xq1 - slope * tq1
        xs = np.array([theo.min(), theo.max()])
        ax.plot(xs, intercept + slope * xs, color=NAVY, linestyle="--", linewidth=1)
        ax.set_xlabel("Normal quantiles", color=MUTED); ax.set_ylabel(feature, color=MUTED)
        _style(ax, "QQ-style normality check")
    else:
        ax.text(0.5, 0.5, "Too few unique values for QQ-style view", ha="center", va="center", color=MUTED)
        ax.axis("off")
    fig.suptitle(f"Selected-feature distribution diagnostic: {feature}", fontsize=16, fontweight="bold", color=NAVY, y=1.02)
    return _save(fig, path)


def plot_qc_artifact_model(path: Path, framework: str = "Auto / all QC") -> Path:
    families = [
        ("Additive\ninterference", "noise, hum, competing speech"),
        ("Gain / level\ndynamics", "AGC, distance, level drift"),
        ("Reverberation\n/ echo", "room tail, boundary blur"),
        ("Channel / device\n/ platform", "filtering, bandwidth, codec"),
        ("Nonlinear\ndistortion", "clipping, saturation"),
        ("Temporal\ndiscontinuities", "dropouts, glitches, jumps"),
    ]
    fig, ax = plt.subplots(figsize=(12.5, 5.7))
    ax.axis("off")
    ax.text(0.5, 0.93, "QC source overview", ha="center", va="center", fontsize=16, fontweight="bold", color=NAVY)
    ax.text(0.5, 0.855, "Manual QC uses human metadata flags; Automated QC uses numeric QC metrics from the uploaded QC table.", ha="center", va="center", fontsize=10.5, color=MUTED)
    xs = np.linspace(0.08, 0.92, len(families))
    for i, ((title, desc), x) in enumerate(zip(families, xs)):
        color = [TEAL, GOLD, "#6B5DD3", "#4E7AA8", RED, "#7A8798"][i]
        rect = plt.Rectangle((x-0.07, 0.36), 0.14, 0.29, transform=ax.transAxes, facecolor=color, alpha=0.95, edgecolor="white", linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x, 0.54, title, ha="center", va="center", fontsize=9.5, fontweight="bold", color="white", transform=ax.transAxes)
        ax.text(x, 0.29, desc, ha="center", va="center", fontsize=8.2, color=MUTED, transform=ax.transAxes, wrap=True)
    source_note = {
        "Manual QC": "Current source: Manual QC. Uses modality-aware human metadata flags such as acquisition validity, audio QC, video QC, and face-visibility QC.",
        "Automated QC": "Current source: Automated QC. Uses the uploaded numeric QC table only; metadata manual flags are not mixed into this mode.",
        "Auto / all QC": "Current source: Automated QC. All detected numeric QC variables are used when available.",
        "Acoustic QC": "Current source: Automated acoustic QC. Uses detected acoustic/audio QC variables when available.",
        "Kinematic QC": "Current source: Automated kinematic QC. Uses detected video/landmark/tracking QC variables when available.",
    }.get(str(framework), f"Current source: {framework}.")
    ax.text(0.5, 0.165, source_note, ha="center", va="center", fontsize=9.5, color=NAVY, transform=ax.transAxes, wrap=True)
    ax.text(0.5, 0.075, "Use QC to ask whether feature variation, missingness, or outliers are plausibly explained by acquisition artifacts before interpreting them as speech physiology.", ha="center", va="center", fontsize=9.5, color=NAVY, transform=ax.transAxes, wrap=True)
    return _save(fig, path)

def plot_qc_family_burden(family_summary: pd.DataFrame, path: Path) -> Path:
    if family_summary is None or family_summary.empty or "artifact_family" not in family_summary.columns:
        return _empty(path, "No QC family summary was available. Load a QC table with qadd/qgain/qrev/qchan/qdist/qtemp/qdrop metrics.", "QC artifact-family burden")
    df = family_summary.copy()
    df["n_qc_metrics"] = pd.to_numeric(df.get("n_qc_metrics", 0), errors="coerce").fillna(0)
    df["median_metric_iqr"] = pd.to_numeric(df.get("median_metric_iqr", np.nan), errors="coerce")
    df["median_row_flag_fraction"] = pd.to_numeric(df.get("median_row_flag_fraction", np.nan), errors="coerce").fillna(0)
    df = df.sort_values("n_qc_metrics", ascending=True)
    fig, ax = plt.subplots(figsize=(11, max(5.2, 0.55 * len(df))))
    vals = df["median_row_flag_fraction"].astype(float)
    colors = [RED if v >= .20 else GOLD if v >= .05 else TEAL for v in vals]
    labels = [f"{fam}  (metrics={int(n)})" for fam, n in zip(df["artifact_family"].astype(str), df["n_qc_metrics"])]
    ax.barh(labels, vals, color=colors)
    ax.set_xlabel("Median fraction of elevated QC metrics per row", color=MUTED)
    ax.set_xlim(0, max(0.05, min(1.0, float(vals.max()) * 1.2 if len(vals) else .1)))
    ax.axvline(.05, color=GOLD, linestyle="--", linewidth=1)
    ax.axvline(.20, color=RED, linestyle="--", linewidth=1)
    _style(ax, "QC burden by artifact family")
    return _save(fig, path)


def plot_qc_metric_distributions(qc_df: pd.DataFrame | None, catalog: pd.DataFrame, path: Path, max_metrics: int = 12) -> Path:
    if qc_df is None or qc_df.empty or catalog is None or catalog.empty:
        return _empty(path, "No QC flags or metrics were available for plotting.", "QC value distributions")
    cat = catalog.copy()
    cat["n_valid"] = pd.to_numeric(cat.get("n_valid", np.nan), errors="coerce")
    cat["n_unique"] = pd.to_numeric(cat.get("n_unique", np.nan), errors="coerce")
    cat = cat[cat["n_valid"].fillna(0) > 0].copy()
    if cat.empty:
        return _empty(path, "All candidate QC variables were missing in this scope.", "QC value distributions")
    cols_all = [c for c in cat["qc_variable"].astype(str).tolist() if c in qc_df.columns]
    if not cols_all:
        return _empty(path, "No catalogued QC variables were present in the QC table.", "QC value distributions")
    families = set(cat.get("artifact_family", pd.Series(dtype=str)).astype(str).tolist())
    is_manual = bool(families) and all(f.startswith("Manual ") for f in families)
    if is_manual:
        rows = []
        for col in cols_all:
            x = pd.to_numeric(qc_df[col], errors="coerce")
            if x.notna().sum() == 0:
                continue
            fam = str(cat.loc[cat["qc_variable"].astype(str).eq(col), "artifact_family"].iloc[0]) if (cat["qc_variable"].astype(str).eq(col)).any() else "Manual QC"
            rows.append({"qc_variable": col, "artifact_family": fam, "flag_rate": float(x.fillna(0).mean()), "n_valid": int(x.notna().sum())})
        df = pd.DataFrame(rows).sort_values("flag_rate", ascending=True).tail(max_metrics)
        if df.empty:
            return _empty(path, "Manual QC flags were present but contained no usable values in this scope.", "Manual QC flag prevalence")
        labels = [f"{v}\n{f}" for v, f in zip(df["qc_variable"].astype(str), df["artifact_family"].astype(str))]
        fig, ax = plt.subplots(figsize=(12, max(5.2, 0.46 * len(df))))
        ax.barh(labels, df["flag_rate"].astype(float), color=TEAL)
        ax.set_xlabel("Flagged fraction of recordings", color=MUTED)
        ax.set_xlim(0, max(0.05, min(1.0, float(df["flag_rate"].max()) * 1.25 if not df.empty else 0.1)))
        _style(ax, "Manual QC flag prevalence")
        return _save(fig, path)
    cat["iqr"] = pd.to_numeric(cat.get("iqr", np.nan), errors="coerce")
    cat = cat.sort_values(["iqr", "n_valid"], ascending=[False, False]).head(max_metrics)
    cols = [c for c in cat["qc_variable"].astype(str).tolist() if c in qc_df.columns and pd.to_numeric(qc_df[c], errors="coerce").notna().any()]
    if not cols:
        return _empty(path, "No non-missing QC variables were available for distribution plotting.", "QC value distributions")
    ncols = 3; nrows = int(np.ceil(len(cols)/ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 3.3*nrows))
    axes = np.array(axes).reshape(-1)
    for ax, col in zip(axes, cols):
        x = pd.to_numeric(qc_df[col], errors="coerce").dropna()
        fam = str(cat.loc[cat["qc_variable"].astype(str).eq(col), "artifact_family"].iloc[0]) if (cat["qc_variable"].astype(str).eq(col)).any() else "QC"
        ax.hist(x, bins=min(26, max(6, int(np.sqrt(len(x))))), color=TEAL, alpha=.88, edgecolor="white")
        ax.axvline(float(x.median()), color=NAVY, linestyle="--", linewidth=1)
        ax.set_title(f"{col}\n{fam}", fontsize=8.5, color=NAVY)
        _style(ax)
    for ax in axes[len(cols):]:
        ax.axis("off")
    fig.suptitle("QC value distributions: highest-spread indicators", fontsize=15, fontweight="bold", color=NAVY, y=1.01)
    return _save(fig, path)


def plot_qc_top_feature_associations(qc_corr: pd.DataFrame, path: Path, top_n: int = 25) -> Path:
    required = {"feature", "qc_variable", "spearman_rho"}
    if qc_corr is None or qc_corr.empty or not required.issubset(qc_corr.columns):
        return _empty(path, "No feature-QC correlations were available. Load an aligned QC table.", "Feature-QC associations")
    df = qc_corr.copy()
    df["abs_rho"] = pd.to_numeric(df["spearman_rho"], errors="coerce").abs()
    df = df.dropna(subset=["abs_rho"]).sort_values("abs_rho", ascending=False).head(top_n).sort_values("abs_rho", ascending=True)
    if df.empty:
        return _empty(path, "No non-missing feature-QC associations were available.", "Feature-QC associations")
    labels = [f"{f}\n× {q}" for f, q in zip(df["feature"].astype(str), df["qc_variable"].astype(str))]
    colors = [RED if r < 0 else TEAL for r in pd.to_numeric(df["spearman_rho"], errors="coerce").fillna(0)]
    fig, ax = plt.subplots(figsize=(12, max(5.5, 0.42*len(df))))
    ax.barh(labels, df["abs_rho"].astype(float), color=colors)
    ax.set_xlabel("Absolute Spearman rho", color=MUTED)
    ax.axvline(.30, color=GOLD, linestyle="--", linewidth=1, label="monitor |rho|=.30")
    ax.axvline(.50, color=RED, linestyle="--", linewidth=1, label="review |rho|=.50")
    _style(ax, "Strongest feature-QC monotonic associations")
    ax.legend(frameon=False, fontsize=8)
    return _save(fig, path)


def plot_qc_feature_association_heatmap(family_assoc: pd.DataFrame, path: Path, top_features: int = 30) -> Path:
    required = {"feature", "artifact_family", "max_abs_spearman"}
    if family_assoc is None or family_assoc.empty or not required.issubset(family_assoc.columns):
        return _empty(path, "No family-level feature-QC association summary was available.", "Feature × QC-family associations")
    df = family_assoc.copy()
    df["max_abs_spearman"] = pd.to_numeric(df["max_abs_spearman"], errors="coerce")
    features = df.groupby("feature")["max_abs_spearman"].max().sort_values(ascending=False).head(top_features).index.tolist()
    mat = df[df["feature"].isin(features)].pivot_table(index="feature", columns="artifact_family", values="max_abs_spearman", aggfunc="max").fillna(0)
    if mat.empty:
        return _empty(path, "No feature-QC-family matrix could be constructed.", "Feature × QC-family associations")
    mat = mat.loc[features]
    fig, ax = plt.subplots(figsize=(10.8, max(6.0, 0.28*len(mat))))
    im = ax.imshow(mat.to_numpy(float), aspect="auto", vmin=0, vmax=max(.5, float(mat.to_numpy(float).max())), cmap="YlGnBu")
    ax.set_xticks(range(len(mat.columns))); ax.set_xticklabels(mat.columns, rotation=35, ha="right", fontsize=8, color=MUTED)
    ax.set_yticks(range(len(mat.index))); ax.set_yticklabels(mat.index, fontsize=7, color=MUTED)
    ax.set_title("Feature sensitivity by QC artifact family", fontsize=14, fontweight="bold", color=NAVY, pad=12)
    cbar = fig.colorbar(im, ax=ax, fraction=.025, pad=.02); cbar.set_label("max |Spearman rho|", color=MUTED); cbar.ax.tick_params(labelsize=8, colors=MUTED)
    return _save(fig, path)


def plot_qc_missingness_associations(missing_assoc: pd.DataFrame, path: Path, top_n: int = 25) -> Path:
    if missing_assoc is None or missing_assoc.empty or "abs_spearman" not in missing_assoc.columns:
        return _empty(path, "No QC-linked missingness associations were available.", "Missingness linked to QC")
    df = missing_assoc.copy().dropna(subset=["abs_spearman"]).sort_values("abs_spearman", ascending=False).head(top_n).sort_values("abs_spearman", ascending=True)
    if df.empty:
        return _empty(path, "No non-missing QC-missingness associations were available.", "Missingness linked to QC")
    labels = [f"{f}\n× {q}" for f, q in zip(df["feature"].astype(str), df["qc_variable"].astype(str))]
    fig, ax = plt.subplots(figsize=(12, max(5.2, 0.42*len(df))))
    ax.barh(labels, df["abs_spearman"].astype(float), color=GOLD)
    ax.set_xlabel("Absolute Spearman rho between feature-missing indicator and QC metric", color=MUTED)
    ax.axvline(.30, color=RED, linestyle="--", linewidth=1)
    _style(ax, "Feature missingness associated with QC metrics")
    return _save(fig, path)


def plot_qc_row_burden(row_burden: pd.DataFrame, path: Path, top_n: int = 35) -> Path:
    if row_burden is None or row_burden.empty or "total_qc_flags" not in row_burden.columns:
        return _empty(path, "No row-level QC burden summary was available.", "Row-level QC burden")
    df = row_burden.copy()
    df["total_qc_flags"] = pd.to_numeric(df["total_qc_flags"], errors="coerce").fillna(0)
    df = df.sort_values("total_qc_flags", ascending=False).head(top_n).sort_values("total_qc_flags", ascending=True)
    if df.empty or df["total_qc_flags"].max() == 0:
        return _empty(path, "No rows exceeded robust QC elevation thresholds. Still review family distributions and feature-QC associations.", "Row-level QC burden")
    label_col = next((c for c in ["file_name", "filename", "source_file", "record_key", "subject_id", "participant_id"] if c in df.columns), "row_index")
    fig, ax = plt.subplots(figsize=(10.8, max(5.2, 0.35*len(df))))
    ax.barh(df[label_col].astype(str), df["total_qc_flags"].astype(float), color=GOLD)
    ax.set_xlabel("Number of elevated QC metrics", color=MUTED)
    _style(ax, "Rows / recordings with highest QC burden")
    return _save(fig, path)


def plot_selected_feature_qc_scatter(feature_df: pd.DataFrame, qc_df: pd.DataFrame | None, feature: str, qc_variable: str, path: Path) -> Path:
    if feature_df is None or feature_df.empty or qc_df is None or qc_df.empty or not feature or not qc_variable:
        return _empty(path, "Select one feature and one QC variable after loading an aligned QC table.", "Selected feature × QC metric")
    # Align by common key if possible, else by row order when lengths match.
    join_keys = [k for k in ["record_key", "file_name", "filename", "source_file", "audio_file", "subject_id", "participant_id"] if k in feature_df.columns and k in qc_df.columns]
    if join_keys:
        merged = feature_df[[join_keys[0], feature]].merge(qc_df[[join_keys[0], qc_variable]], on=join_keys[0], how="inner")
    elif len(feature_df) == len(qc_df):
        merged = pd.DataFrame({feature: pd.to_numeric(feature_df[feature], errors="coerce"), qc_variable: pd.to_numeric(qc_df[qc_variable], errors="coerce")})
    else:
        return _empty(path, "QC table could not be aligned to the feature table for selected scatter plotting.", "Selected feature × QC metric")
    x = pd.to_numeric(merged[qc_variable], errors="coerce")
    y = pd.to_numeric(merged[feature], errors="coerce")
    pair = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(pair) < 4:
        return _empty(path, "Too few paired non-missing observations for this feature/QC pair.", "Selected feature × QC metric")
    rho = pair["x"].corr(pair["y"], method="spearman") if pair["x"].nunique() > 1 and pair["y"].nunique() > 1 else np.nan
    fig, ax = plt.subplots(figsize=(9.5, 6.0))
    ax.scatter(pair["x"], pair["y"], s=28, alpha=.72, color=TEAL, edgecolor="white", linewidth=.35)
    ax.set_xlabel(qc_variable, color=MUTED)
    ax.set_ylabel(feature, color=MUTED)
    title = f"{feature} vs QC: {qc_variable}"
    _style(ax, title)
    if pd.notna(rho):
        ax.text(.02, .98, f"Spearman ρ = {rho:.2f}\nn = {len(pair)}", transform=ax.transAxes, va="top", ha="left", fontsize=10, color=NAVY, bbox=dict(facecolor="white", edgecolor=GRID, boxstyle="round,pad=.35"))
    return _save(fig, path)


# -----------------------------------------------------------------------------
# Feature relationship plots
# -----------------------------------------------------------------------------

def _corr_matrix_from_long(corr_long: pd.DataFrame, top_n: int = 60) -> pd.DataFrame:
    if corr_long is None or corr_long.empty or not {"feature_1","feature_2","spearman_rho"}.issubset(corr_long.columns):
        return pd.DataFrame()
    df = corr_long.copy()
    df["abs_spearman"] = pd.to_numeric(df.get("abs_spearman", df["spearman_rho"].abs()), errors="coerce")
    features = df.groupby("feature_1")["abs_spearman"].max().combine_first(df.groupby("feature_2")["abs_spearman"].max()).sort_values(ascending=False).head(top_n).index.tolist()
    if not features:
        features = sorted(set(df["feature_1"].astype(str)).union(set(df["feature_2"].astype(str))))[:top_n]
    mat = pd.DataFrame(np.eye(len(features)), index=features, columns=features)
    for _, r in df.iterrows():
        a, b = str(r["feature_1"]), str(r["feature_2"])
        if a in mat.index and b in mat.columns:
            val = pd.to_numeric(r["spearman_rho"], errors="coerce")
            mat.loc[a,b] = val; mat.loc[b,a] = val
    return mat


def plot_relationship_correlation_heatmap(corr_long: pd.DataFrame, path: Path, top_n: int = 60) -> Path:
    mat = _corr_matrix_from_long(corr_long, top_n=top_n)
    if mat.empty:
        return _empty(path, "At least two numeric feature columns are required for feature-feature correlation review.", "Feature correlation heatmap")
    order = mat.abs().sum(axis=1).sort_values(ascending=False).index.tolist()
    mat = mat.loc[order, order]
    fig, ax = plt.subplots(figsize=(11.5, 10.2))
    im = ax.imshow(mat.to_numpy(float), vmin=-1, vmax=1, cmap="coolwarm", aspect="auto")
    step = max(1, len(mat)//35)
    ticks = list(range(0, len(mat), step))
    ax.set_xticks(ticks); ax.set_xticklabels([mat.columns[i] for i in ticks], rotation=90, fontsize=6, color=MUTED)
    ax.set_yticks(ticks); ax.set_yticklabels([mat.index[i] for i in ticks], fontsize=6, color=MUTED)
    ax.set_title("Feature-feature Spearman correlation structure", fontsize=14, fontweight="bold", color=NAVY, pad=12)
    cbar = fig.colorbar(im, ax=ax, fraction=.025, pad=.02); cbar.set_label("Spearman rho", color=MUTED); cbar.ax.tick_params(labelsize=8, colors=MUTED)
    return _save(fig, path)


def plot_relationship_redundant_pairs(pairs: pd.DataFrame, path: Path, top_n: int = 30) -> Path:
    if pairs is None or pairs.empty or "abs_spearman" not in pairs.columns:
        return _empty(path, "No feature pairs exceeded the redundancy threshold. This is good, but still inspect PCA and family structure.", "Top redundant pairs")
    df = pairs.copy().sort_values("abs_spearman", ascending=False).head(top_n).sort_values("abs_spearman", ascending=True)
    labels = [f"{a}\n× {b}" for a,b in zip(df["feature_1"].astype(str), df["feature_2"].astype(str))]
    fig, ax = plt.subplots(figsize=(12, max(5.5, .45*len(df))))
    colors = [RED if v >= .90 else GOLD for v in df["abs_spearman"].astype(float)]
    ax.barh(labels, df["abs_spearman"].astype(float), color=colors)
    ax.axvline(.80, color=GOLD, linestyle="--", linewidth=1)
    ax.axvline(.90, color=RED, linestyle="--", linewidth=1)
    ax.set_xlim(0,1); ax.set_xlabel("Absolute Spearman rho", color=MUTED)
    _style(ax, "Strongly redundant feature pairs")
    return _save(fig, path)


def plot_relationship_family_matrix(family_matrix: pd.DataFrame, path: Path) -> Path:
    if family_matrix is None or family_matrix.empty or not {"family_1","family_2","mean_abs_spearman"}.issubset(family_matrix.columns):
        return _empty(path, "Family/block relationship matrix unavailable. Load a feature registry for stronger subsystem labels.", "Feature-family correlation matrix")
    fams = sorted(set(family_matrix["family_1"].astype(str)).union(set(family_matrix["family_2"].astype(str))))
    mat = pd.DataFrame(0.0, index=fams, columns=fams)
    for _, r in family_matrix.iterrows():
        a,b = str(r["family_1"]), str(r["family_2"])
        v = float(r["mean_abs_spearman"])
        mat.loc[a,b] = v; mat.loc[b,a] = v
    fig, ax = plt.subplots(figsize=(9.6, 7.8))
    im = ax.imshow(mat.to_numpy(float), vmin=0, vmax=max(.5, float(mat.to_numpy(float).max())), cmap="YlGnBu")
    ax.set_xticks(range(len(fams))); ax.set_xticklabels(fams, rotation=35, ha="right", fontsize=9, color=MUTED)
    ax.set_yticks(range(len(fams))); ax.set_yticklabels(fams, fontsize=9, color=MUTED)
    ax.set_title("Mean absolute feature correlation by family", fontsize=14, fontweight="bold", color=NAVY, pad=12)
    cbar = fig.colorbar(im, ax=ax, fraction=.03, pad=.02); cbar.set_label("Mean |rho|", color=MUTED); cbar.ax.tick_params(labelsize=8, colors=MUTED)
    return _save(fig, path)


def plot_relationship_pca_scree(pca_summary: pd.DataFrame, path: Path) -> Path:
    if pca_summary is None or pca_summary.empty or "variance_percent" not in pca_summary.columns:
        return _empty(path, "PCA requires at least two usable numeric features with enough non-missing observations.", "PCA scree")
    df = pca_summary.copy().head(12)
    x = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    ax.bar(x, df["variance_percent"].astype(float), color=TEAL)
    ax.plot(x, df["cumulative_variance_percent"].astype(float), marker="o", color=NAVY, linewidth=2)
    ax.set_xticks(x); ax.set_xticklabels(df["component"].astype(str), color=MUTED)
    ax.set_ylabel("Variance explained (%)", color=MUTED)
    ax.set_ylim(0, max(100, float(df["cumulative_variance_percent"].max())*1.05))
    _style(ax, "PCA scree: unsupervised feature-space dimensionality")
    return _save(fig, path)


def plot_relationship_pca_scores(pca_scores: pd.DataFrame, feature_df: pd.DataFrame | None, path: Path) -> Path:
    if pca_scores is None or pca_scores.empty or not {"PC1","PC2"}.issubset(pca_scores.columns):
        return _empty(path, "PCA scores unavailable for this dataset.", "PCA recording map")
    fig, ax = plt.subplots(figsize=(9.2, 6.6))
    x = pd.to_numeric(pca_scores["PC1"], errors="coerce"); y = pd.to_numeric(pca_scores["PC2"], errors="coerce")
    ax.scatter(x, y, s=28, alpha=.75, color=TEAL, edgecolor="white", linewidth=.35)
    ax.axhline(0, color=GRID, linewidth=1); ax.axvline(0, color=GRID, linewidth=1)
    ax.set_xlabel("PC1 score", color=MUTED); ax.set_ylabel("PC2 score", color=MUTED)
    _style(ax, "Recording map in unsupervised feature space")
    return _save(fig, path)


def plot_relationship_pca_loadings(loadings: pd.DataFrame, path: Path, top_n: int = 18) -> Path:
    if loadings is None or loadings.empty or not {"component","feature","loading"}.issubset(loadings.columns):
        return _empty(path, "PCA loadings unavailable for this dataset.", "PCA top loadings")
    df = loadings.copy()
    df = df[df["component"].isin(["PC1","PC2","PC3"])].copy()
    if df.empty:
        return _empty(path, "No early-component loadings available.", "PCA top loadings")
    # top across first 3 components
    df = df.sort_values("abs_loading", ascending=False).head(top_n).sort_values("abs_loading", ascending=True)
    labels = [f"{c}: {f}" for c,f in zip(df["component"].astype(str), df["feature"].astype(str))]
    colors = [TEAL if v >= 0 else GOLD for v in df["loading"].astype(float)]
    fig, ax = plt.subplots(figsize=(11.5, max(5.6, .38*len(df))))
    ax.barh(labels, df["loading"].astype(float), color=colors)
    ax.axvline(0, color=GRID, linewidth=1)
    ax.set_xlabel("PCA loading", color=MUTED)
    _style(ax, "Largest PCA loadings across early components")
    return _save(fig, path)



def plot_relationship_dimensionality_profile(pca_summary: pd.DataFrame, loadings: pd.DataFrame, path: Path, top_n: int = 12) -> Path:
    """Combined PCA scree and leading-loading panel for Feature Relationships."""
    if pca_summary is None or pca_summary.empty or "variance_percent" not in pca_summary.columns:
        return _empty(path, "PCA requires at least two usable numeric features with enough non-missing observations.", "Dimensionality profile")
    ps = pca_summary.copy().head(8)
    fig = plt.figure(figsize=(12.5, 7.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.35], wspace=.32)
    ax1 = fig.add_subplot(gs[0,0])
    x = np.arange(len(ps))
    ax1.bar(x, pd.to_numeric(ps["variance_percent"], errors="coerce"), color=TEAL)
    if "cumulative_variance_percent" in ps.columns:
        ax1.plot(x, pd.to_numeric(ps["cumulative_variance_percent"], errors="coerce"), marker="o", color=NAVY, linewidth=2)
    ax1.set_xticks(x); ax1.set_xticklabels(ps.get("component", pd.Series([f"PC{i+1}" for i in x])).astype(str), color=MUTED)
    ax1.set_ylabel("Variance explained (%)", color=MUTED)
    ax1.set_ylim(0, max(100, float(pd.to_numeric(ps.get("cumulative_variance_percent", ps["variance_percent"]), errors="coerce").max())*1.05))
    _style(ax1, "Variance concentration")

    ax2 = fig.add_subplot(gs[0,1])
    if loadings is None or loadings.empty or not {"component","feature","loading"}.issubset(loadings.columns):
        ax2.text(.5, .5, "PCA loadings unavailable", ha="center", va="center", color=MUTED)
        ax2.axis("off")
    else:
        df = loadings.copy()
        if "abs_loading" not in df.columns:
            df["abs_loading"] = pd.to_numeric(df["loading"], errors="coerce").abs()
        df = df[df["component"].astype(str).isin(["PC1", "PC2", "PC3"])].dropna(subset=["abs_loading"])
        df = df.sort_values("abs_loading", ascending=False).head(top_n).sort_values("abs_loading", ascending=True)
        if df.empty:
            ax2.text(.5, .5, "No early-component loading drivers", ha="center", va="center", color=MUTED)
            ax2.axis("off")
        else:
            labels = [f"{c}: {f}" for c, f in zip(df["component"].astype(str), df["feature"].astype(str))]
            vals = pd.to_numeric(df["loading"], errors="coerce")
            colors = [TEAL if v >= 0 else GOLD for v in vals]
            ax2.barh(labels, vals, color=colors)
            ax2.axvline(0, color=GRID, linewidth=1)
            ax2.set_xlabel("PCA loading", color=MUTED)
            _style(ax2, "Largest early-component drivers")
    fig.suptitle("Dimensionality profile: variance concentration and leading feature drivers", fontsize=15, fontweight="bold", color=NAVY, y=.98)
    return _save(fig, path)

def plot_selected_feature_correlations(corr_long: pd.DataFrame, feature: str, path: Path, top_n: int = 30) -> Path:
    if corr_long is None or corr_long.empty or not feature:
        return _empty(path, "Select a feature after running relationship analysis.", "Selected feature correlations")
    df = corr_long[(corr_long["feature_1"].astype(str)==feature) | (corr_long["feature_2"].astype(str)==feature)].copy()
    if df.empty:
        return _empty(path, f"No pairwise correlations involving {feature} were available.", "Selected feature correlations")
    df["other_feature"] = np.where(df["feature_1"].astype(str)==feature, df["feature_2"].astype(str), df["feature_1"].astype(str))
    df = df.dropna(subset=["abs_spearman"]).sort_values("abs_spearman", ascending=False).head(top_n).sort_values("abs_spearman", ascending=True)
    fig, ax = plt.subplots(figsize=(11, max(5.5, .36*len(df))))
    colors = [RED if abs(v) >= .80 else GOLD if abs(v) >= .60 else TEAL for v in df["spearman_rho"].astype(float)]
    ax.barh(df["other_feature"].astype(str), df["spearman_rho"].astype(float), color=colors)
    ax.axvline(0, color=GRID, linewidth=1)
    ax.axvline(.80, color=RED, linestyle="--", linewidth=1, alpha=.8); ax.axvline(-.80, color=RED, linestyle="--", linewidth=1, alpha=.8)
    ax.set_xlabel("Spearman rho", color=MUTED); ax.set_xlim(-1,1)
    _style(ax, f"Strongest feature links for {feature}")
    return _save(fig, path)

# Group / outcome screening plots

def plot_screening_group_balance(balance: pd.DataFrame, path: Path) -> Path:
    required = {"group_variable", "level", "n_rows"}
    if balance is None or balance.empty or not required.issubset(balance.columns):
        return _empty(path, "No categorical group/outcome variables were available for balance plotting.", "Group balance")
    df = balance.copy()
    # pick the most informative variable: not too many levels, largest total rows
    totals = df.groupby("group_variable")["n_rows"].sum().sort_values(ascending=False)
    group = str(totals.index[0])
    sub = df[df["group_variable"].astype(str).eq(group)].copy().sort_values("n_rows", ascending=True)
    sub = sub.tail(30)
    fig, ax = plt.subplots(figsize=(10.5, max(4.8, 0.36 * len(sub))))
    colors = [RED if float(n) < 10 else GOLD if float(n) < 20 else TEAL for n in sub["n_rows"]]
    ax.barh(sub["level"].astype(str), pd.to_numeric(sub["n_rows"], errors="coerce"), color=colors)
    ax.set_xlabel("Rows / recordings", color=MUTED)
    _style(ax, f"Group balance: {group}")
    return _save(fig, path)


def plot_screening_effect_ranking(cont: pd.DataFrame, cat: pd.DataFrame, path: Path, top_n: int = 30) -> Path:
    frames=[]
    if cont is not None and not cont.empty:
        a = cont.copy(); a["screening_source"] = a.get("outcome_variable", "continuous").astype(str); a["label"] = a["feature"].astype(str) + " → " + a["screening_source"].astype(str); frames.append(a)
    if cat is not None and not cat.empty:
        b = cat.copy(); b["screening_source"] = b.get("group_variable", "group").astype(str); b["label"] = b["feature"].astype(str) + " ↔ " + b["screening_source"].astype(str); frames.append(b)
    if not frames:
        return _empty(path, "No feature-outcome or feature-group screening effects were available.", "Screening effects")
    df = pd.concat(frames, ignore_index=True, sort=False)
    df["abs_effect"] = pd.to_numeric(df["abs_effect"], errors="coerce")
    df = df.dropna(subset=["abs_effect"]).sort_values("abs_effect", ascending=False).head(top_n).sort_values("abs_effect")
    if df.empty:
        return _empty(path, "Screening effects were not evaluable for the current dataset.", "Screening effects")
    fig, ax = plt.subplots(figsize=(11.5, max(5, 0.34 * len(df))))
    colors = [RED if v >= .50 else GOLD if v >= .30 else TEAL for v in df["abs_effect"]]
    ax.barh(df["label"].astype(str), df["abs_effect"], color=colors)
    ax.set_xlabel("Absolute descriptive effect size", color=MUTED)
    ax.axvline(.30, color=GOLD, linestyle="--", linewidth=1)
    ax.axvline(.50, color=RED, linestyle="--", linewidth=1)
    _style(ax, "Top descriptive feature screening effects")
    return _save(fig, path)


def plot_screening_continuous_heatmap(cont: pd.DataFrame, path: Path, max_features: int = 40, max_outcomes: int = 12) -> Path:
    if cont is None or cont.empty or not {"feature", "outcome_variable", "effect"}.issubset(cont.columns):
        return _empty(path, "No continuous outcome screening results were available.", "Feature × continuous outcome")
    df = cont.copy()
    df["abs_effect"] = pd.to_numeric(df["abs_effect"], errors="coerce")
    df["effect"] = pd.to_numeric(df["effect"], errors="coerce")
    top_feats = df.groupby("feature")["abs_effect"].max().sort_values(ascending=False).head(max_features).index.tolist()
    top_out = df.groupby("outcome_variable")["abs_effect"].max().sort_values(ascending=False).head(max_outcomes).index.tolist()
    mat = df[df["feature"].isin(top_feats) & df["outcome_variable"].isin(top_out)].pivot_table(index="feature", columns="outcome_variable", values="effect", aggfunc="max")
    if mat.empty:
        return _empty(path, "No evaluable continuous outcome heatmap could be formed.", "Feature × continuous outcome")
    fig, ax = plt.subplots(figsize=(max(8, 0.5 * len(mat.columns) + 5), max(6, 0.24 * len(mat.index) + 2)))
    im = ax.imshow(mat.fillna(0).to_numpy(dtype=float), vmin=-1, vmax=1, cmap="coolwarm", aspect="auto")
    ax.set_xticks(range(len(mat.columns))); ax.set_xticklabels(mat.columns, rotation=45, ha="right", fontsize=8, color=MUTED)
    ax.set_yticks(range(len(mat.index))); ax.set_yticklabels(mat.index, fontsize=6, color=MUTED)
    ax.set_title("Feature × continuous outcome Spearman screen", fontsize=14, fontweight="bold", color=NAVY, pad=12)
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02); cbar.set_label("Spearman rho", color=MUTED); cbar.ax.tick_params(labelsize=8, colors=MUTED)
    return _save(fig, path)


def plot_screening_group_heatmap(cat: pd.DataFrame, path: Path, max_features: int = 40, max_groups: int = 12) -> Path:
    if cat is None or cat.empty or not {"feature", "group_variable", "effect"}.issubset(cat.columns):
        return _empty(path, "No categorical group screening results were available.", "Feature × group")
    df = cat.copy(); df["abs_effect"] = pd.to_numeric(df["abs_effect"], errors="coerce"); df["effect"] = pd.to_numeric(df["effect"], errors="coerce")
    top_feats = df.groupby("feature")["abs_effect"].max().sort_values(ascending=False).head(max_features).index.tolist()
    top_groups = df.groupby("group_variable")["abs_effect"].max().sort_values(ascending=False).head(max_groups).index.tolist()
    mat = df[df["feature"].isin(top_feats) & df["group_variable"].isin(top_groups)].pivot_table(index="feature", columns="group_variable", values="effect", aggfunc="max")
    if mat.empty:
        return _empty(path, "No evaluable group screening heatmap could be formed.", "Feature × group")
    lim = float(np.nanmax(np.abs(mat.to_numpy(dtype=float)))) if mat.size else 1.0
    lim = max(lim, 1.0)
    fig, ax = plt.subplots(figsize=(max(8, 0.5 * len(mat.columns) + 5), max(6, 0.24 * len(mat.index) + 2)))
    im = ax.imshow(mat.fillna(0).to_numpy(dtype=float), vmin=-lim, vmax=lim, cmap="coolwarm", aspect="auto")
    ax.set_xticks(range(len(mat.columns))); ax.set_xticklabels(mat.columns, rotation=45, ha="right", fontsize=8, color=MUTED)
    ax.set_yticks(range(len(mat.index))); ax.set_yticklabels(mat.index, fontsize=6, color=MUTED)
    ax.set_title("Feature × categorical group contrast screen", fontsize=14, fontweight="bold", color=NAVY, pad=12)
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02); cbar.set_label("Signed robust effect", color=MUTED); cbar.ax.tick_params(labelsize=8, colors=MUTED)
    return _save(fig, path)


def plot_screening_effect_landscape(cont: pd.DataFrame, cat: pd.DataFrame, path: Path) -> Path:
    frames=[]
    if cont is not None and not cont.empty:
        a = cont.copy(); a["type"] = "continuous"; frames.append(a)
    if cat is not None and not cat.empty:
        b = cat.copy(); b["type"] = "categorical"; frames.append(b)
    if not frames:
        return _empty(path, "No screening effects were available for the landscape plot.", "Screening landscape")
    df = pd.concat(frames, ignore_index=True, sort=False)
    df["abs_effect"] = pd.to_numeric(df["abs_effect"], errors="coerce")
    df["n_pairwise"] = pd.to_numeric(df["n_pairwise"], errors="coerce")
    df = df.dropna(subset=["abs_effect", "n_pairwise"])
    if df.empty:
        return _empty(path, "Screening effect sizes were not evaluable.", "Screening landscape")
    fig, ax = plt.subplots(figsize=(10.8, 6.3))
    for t, color in [("continuous", TEAL), ("categorical", GOLD)]:
        sub = df[df["type"].eq(t)]
        if not sub.empty:
            ax.scatter(sub["n_pairwise"], sub["abs_effect"], s=28, alpha=0.78, color=color, label=t)
    ax.axhline(.30, color=GOLD, linestyle="--", linewidth=1)
    ax.axhline(.50, color=RED, linestyle="--", linewidth=1)
    ax.set_xlabel("Pairwise N", color=MUTED)
    ax.set_ylabel("Absolute descriptive effect size", color=MUTED)
    ax.legend(frameon=False, fontsize=9)
    _style(ax, "Screening effect landscape")
    return _save(fig, path)


def plot_selected_feature_outcome(df: pd.DataFrame, feature: str, outcome: str, path: Path) -> Path:
    if df is None or df.empty or feature not in df.columns or outcome not in df.columns:
        return _empty(path, "Selected feature or outcome/group variable was not available.", "Selected feature screening")
    x = pd.to_numeric(df[feature], errors="coerce")
    y_raw = df[outcome]
    tmp = pd.DataFrame({"feature": x, "outcome": y_raw}).dropna()
    if tmp.empty:
        return _empty(path, "No paired non-missing values were available for the selected feature and outcome/group.", "Selected feature screening")
    fig, ax = plt.subplots(figsize=(10.8, 6.2))
    if pd.api.types.is_numeric_dtype(y_raw) and tmp["outcome"].nunique(dropna=True) > 10:
        y = pd.to_numeric(tmp["outcome"], errors="coerce")
        tmp = pd.DataFrame({"feature": tmp["feature"], "outcome": y}).dropna()
        ax.scatter(tmp["outcome"], tmp["feature"], s=32, color=TEAL, alpha=0.78)
        if len(tmp) >= 5:
            try:
                z = np.polyfit(tmp["outcome"].astype(float), tmp["feature"].astype(float), deg=1)
                xx = np.linspace(tmp["outcome"].min(), tmp["outcome"].max(), 80)
                ax.plot(xx, z[0]*xx + z[1], color=NAVY, linewidth=2, alpha=.8)
            except Exception:
                pass
        ax.set_xlabel(outcome, color=MUTED); ax.set_ylabel(feature, color=MUTED)
        _style(ax, f"{feature} vs {outcome}")
    else:
        tmp["outcome"] = tmp["outcome"].astype(str)
        levels = tmp["outcome"].value_counts().head(12).index.tolist()
        data = [tmp.loc[tmp["outcome"].eq(level), "feature"].dropna().to_numpy(dtype=float) for level in levels]
        ax.boxplot(data, labels=levels, vert=True, patch_artist=True, boxprops=dict(facecolor="#DDF6F4", color=TEAL), medianprops=dict(color=NAVY), whiskerprops=dict(color=MUTED), capprops=dict(color=MUTED))
        for i, vals in enumerate(data, start=1):
            if len(vals):
                jitter = np.linspace(-0.12, 0.12, len(vals)) if len(vals) < 30 else np.random.default_rng(42).normal(0, .05, len(vals))
                ax.scatter(np.full(len(vals), i) + jitter, vals, s=18, color=GOLD, alpha=.45)
        ax.set_xlabel(outcome, color=MUTED); ax.set_ylabel(feature, color=MUTED)
        ax.tick_params(axis="x", rotation=30)
        _style(ax, f"{feature} by {outcome}")
    return _save(fig, path)


def plot_reliability_status_counts(repeatability: pd.DataFrame, path: Path) -> Path:
    if repeatability is None or repeatability.empty or "reliability_status" not in repeatability.columns:
        return _empty(path, "No repeatability results were available.", "Reliability status")
    counts = repeatability["reliability_status"].astype(str).value_counts().reindex(["stable", "moderate", "variable", "unstable", "not_evaluable"]).dropna()
    if counts.empty:
        return _empty(path, "No repeatability status values were available.", "Reliability status")
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    ax.bar(counts.index, counts.values, color=TEAL, edgecolor=NAVY, linewidth=.8)
    ax.set_ylabel("Number of features", color=MUTED)
    ax.tick_params(axis="x", rotation=20)
    _style(ax, "Feature repeatability status")
    for i, v in enumerate(counts.values):
        ax.text(i, v, str(int(v)), ha="center", va="bottom", color=NAVY, fontweight="bold")
    return _save(fig, path)


def plot_reliability_icc_ranking(repeatability: pd.DataFrame, path: Path, top_n: int = 30) -> Path:
    if repeatability is None or repeatability.empty or "icc1_proxy" not in repeatability.columns:
        return _empty(path, "No ICC-style repeatability estimates were available.", "ICC ranking")
    df = repeatability.copy()
    df["icc1_proxy"] = pd.to_numeric(df["icc1_proxy"], errors="coerce")
    df = df.dropna(subset=["icc1_proxy"]).sort_values("icc1_proxy", ascending=False).head(top_n)
    if df.empty:
        return _empty(path, "No evaluable features had ICC-style estimates.", "ICC ranking")
    fig, ax = plt.subplots(figsize=(10, max(5.2, .32*len(df))))
    ax.barh(df["feature"].astype(str), df["icc1_proxy"], color=TEAL, edgecolor=NAVY, linewidth=.6)
    ax.axvline(.75, color=NAVY, linestyle="--", linewidth=1, label="stable (.75)")
    ax.axvline(.50, color=GOLD, linestyle="--", linewidth=1, label="moderate (.50)")
    ax.axvline(.25, color=RED, linestyle="--", linewidth=1, label="variable (.25)")
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("ICC(1)-style variance-ratio proxy", color=MUTED)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    _style(ax, "Most repeatable features")
    return _save(fig, path)


def plot_reliability_variance_landscape(repeatability: pd.DataFrame, path: Path) -> Path:
    if repeatability is None or repeatability.empty:
        return _empty(path, "No variance-component repeatability table was available.", "Repeatability landscape")
    df = repeatability.copy()
    df["within_subject_variance"] = pd.to_numeric(df.get("within_subject_variance"), errors="coerce")
    df["between_subject_variance"] = pd.to_numeric(df.get("between_subject_variance"), errors="coerce")
    df["icc1_proxy"] = pd.to_numeric(df.get("icc1_proxy"), errors="coerce")
    df = df.dropna(subset=["within_subject_variance", "between_subject_variance", "icc1_proxy"])
    if df.empty:
        return _empty(path, "No evaluable within/between-subject variance estimates were available.", "Repeatability landscape")
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    sizes = 35 + 110 * df["icc1_proxy"].clip(0, 1).fillna(0)
    ax.scatter(df["within_subject_variance"], df["between_subject_variance"], s=sizes, color=TEAL, alpha=.72, edgecolor=NAVY, linewidth=.4)
    ax.set_xscale("symlog", linthresh=1e-6)
    ax.set_yscale("symlog", linthresh=1e-6)
    ax.set_xlabel("Within-subject variance", color=MUTED)
    ax.set_ylabel("Between-subject variance", color=MUTED)
    _style(ax, "Within- vs between-subject variance")
    top = df.sort_values("icc1_proxy", ascending=False).head(8)
    for _, r in top.iterrows():
        ax.text(r["within_subject_variance"], r["between_subject_variance"], str(r["feature"])[:22], fontsize=7, color=NAVY)
    return _save(fig, path)


def plot_reliability_family_summary(family: pd.DataFrame, path: Path) -> Path:
    if family is None or family.empty or "median_icc1_proxy" not in family.columns:
        return _empty(path, "No family-level repeatability summary was available.", "Reliability by family")
    df = family.copy()
    df["median_icc1_proxy"] = pd.to_numeric(df["median_icc1_proxy"], errors="coerce")
    df = df.dropna(subset=["median_icc1_proxy"]).sort_values("median_icc1_proxy", ascending=True)
    if df.empty:
        return _empty(path, "Family-level ICC estimates were not evaluable.", "Reliability by family")
    fig, ax = plt.subplots(figsize=(10, max(4.8, .42*len(df))))
    ax.barh(df["family_or_subsystem"].astype(str), df["median_icc1_proxy"], color=GOLD, edgecolor=NAVY, linewidth=.7)
    ax.axvline(.75, color=NAVY, linestyle="--", linewidth=1)
    ax.axvline(.50, color=TEAL, linestyle="--", linewidth=1)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Median ICC(1)-style proxy", color=MUTED)
    _style(ax, "Repeatability by feature family")
    return _save(fig, path)


def plot_reliability_subject_counts(counts: pd.DataFrame, path: Path) -> Path:
    if counts is None or counts.empty or "n_records" not in counts.columns:
        return _empty(path, "No subject-level record counts were available. A subject identifier is required.", "Subject recording counts")
    df = counts.copy().sort_values("n_records", ascending=False).head(60)
    fig, ax = plt.subplots(figsize=(11, 5.8))
    ax.bar(range(len(df)), pd.to_numeric(df["n_records"], errors="coerce"), color=TEAL, edgecolor=NAVY, linewidth=.5)
    ax.axhline(2, color=GOLD, linestyle="--", linewidth=1)
    ax.set_xlabel("Subjects ranked by record count", color=MUTED)
    ax.set_ylabel("Records per subject", color=MUTED)
    _style(ax, "Repeated-record support by subject")
    return _save(fig, path)


def plot_selected_feature_reliability(df: pd.DataFrame, feature: str, path: Path, subject_col: str | None = None, session_col: str | None = None) -> Path:
    if df is None or df.empty or feature not in df.columns:
        return _empty(path, "Selected feature was not available.", "Selected feature repeatability")
    # Infer subject/session locally if not supplied.
    def _find(names):
        norm = {str(c).lower().replace(" ", "_").replace("-", "_"): c for c in df.columns}
        for n in names:
            if n in norm:
                return norm[n]
        return None
    subject_col = subject_col or _find(["subject_id", "participant_id", "patient_id", "speaker_id"])
    session_col = session_col or _find(["session_id", "visit_id", "clinical_visit_id", "timepoint", "session", "visit"])
    y = pd.to_numeric(df[feature], errors="coerce")
    if not subject_col or subject_col not in df.columns:
        tmp = pd.DataFrame({"value": y}).dropna()
        if tmp.empty:
            return _empty(path, "No valid values were available for the selected feature.", "Selected feature repeatability")
        fig, ax = plt.subplots(figsize=(10, 5.8))
        ax.scatter(np.arange(len(tmp)), tmp["value"], s=24, color=TEAL, alpha=.75)
        ax.set_xlabel("Record index", color=MUTED); ax.set_ylabel(feature, color=MUTED)
        _style(ax, f"{feature}: no subject column detected")
        return _save(fig, path)
    tmp = pd.DataFrame({"subject": df[subject_col].astype(str), "value": y})
    if session_col and session_col in df.columns:
        tmp["session"] = df[session_col].astype(str)
    else:
        tmp["session"] = tmp.groupby("subject").cumcount().astype(str)
    tmp = tmp.dropna(subset=["subject", "value"])
    counts = tmp["subject"].value_counts()
    keep = counts[counts >= 2].head(40).index.tolist()
    tmp = tmp[tmp["subject"].isin(keep)] if keep else tmp.head(80)
    if tmp.empty:
        return _empty(path, "No repeated subjects with valid selected-feature values were available.", "Selected feature repeatability")
    # Create an ordinal x per subject to avoid fragile session sorting.
    tmp["order"] = tmp.groupby("subject").cumcount()
    fig, ax = plt.subplots(figsize=(11, 6.2))
    for sid, sub in tmp.groupby("subject"):
        sub = sub.sort_values("order")
        ax.plot(sub["order"], sub["value"], marker="o", markersize=3.5, linewidth=1.0, alpha=.45, color=TEAL)
    ax.set_xlabel("Repeated record order within subject", color=MUTED)
    ax.set_ylabel(feature, color=MUTED)
    _style(ax, f"{feature}: within-subject repeatability")
    return _save(fig, path)


def plot_recommendation_counts(recs: pd.DataFrame, path: Path) -> Path:
    if recs is None or recs.empty or "readiness_recommendation" not in recs.columns:
        return _empty(path, "No feature recommendation table was available.", "Feature recommendations")
    order = ["recommended", "recommended_with_caution", "review_before_use", "exclude_or_recompute", "exclude_by_default"]
    s = recs["readiness_recommendation"].astype(str).value_counts().reindex(order).fillna(0).astype(int)
    colors = [TEAL, "#8CCB88", GOLD, "#E58E26", RED]
    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    ax.barh([x.replace("_", " ") for x in s.index], s.values, color=colors)
    ax.invert_yaxis(); ax.set_xlabel("Number of features", color=MUTED)
    _style(ax, "Integrated feature readiness categories")
    for i, v in enumerate(s.values):
        ax.text(v + max(1, s.max() * .02), i, str(v), va="center", color=NAVY, fontweight="bold")
    return _save(fig, path)


def plot_recommendation_score_landscape(recs: pd.DataFrame, path: Path) -> Path:
    if recs is None or recs.empty or "readiness_score" not in recs.columns:
        return _empty(path, "No readiness scores were available.", "Feature readiness")
    df = recs.copy()
    df["readiness_score"] = pd.to_numeric(df["readiness_score"], errors="coerce")
    df["missing_fraction"] = pd.to_numeric(df.get("missing_fraction", np.nan), errors="coerce")
    df["max_abs_qc_spearman"] = pd.to_numeric(df.get("max_abs_qc_spearman", np.nan), errors="coerce")
    df = df.dropna(subset=["readiness_score"])
    if df.empty:
        return _empty(path, "No numeric readiness scores were available.", "Feature readiness")
    fig, ax = plt.subplots(figsize=(10, 6))
    x = df["missing_fraction"].fillna(0)
    y = df["readiness_score"]
    qc = df["max_abs_qc_spearman"].fillna(0)
    sizes = 30 + 240 * qc.clip(0, 1)
    colors = df["readiness_recommendation"].map({"recommended": TEAL, "recommended_with_caution": "#8CCB88", "review_before_use": GOLD, "exclude_or_recompute": "#E58E26", "exclude_by_default": RED}).fillna(MUTED)
    ax.scatter(x, y, s=sizes, c=colors, alpha=.82, edgecolor="white", linewidth=.8)
    ax.axhline(80, color=TEAL, linestyle="--", linewidth=1, alpha=.65)
    ax.axhline(60, color=GOLD, linestyle="--", linewidth=1, alpha=.65)
    ax.axhline(40, color=RED, linestyle="--", linewidth=1, alpha=.65)
    ax.set_xlim(-.02, 1.02); ax.set_ylim(-2, 102)
    ax.set_xlabel("Missing fraction", color=MUTED)
    ax.set_ylabel("Integrated readiness score", color=MUTED)
    _style(ax, "Feature readiness landscape")
    ax.text(.01, 83, "recommended zone", color=TEAL, fontsize=9)
    ax.text(.01, 63, "caution/review boundary", color=GOLD, fontsize=9)
    ax.text(.01, 43, "review/exclude boundary", color=RED, fontsize=9)
    return _save(fig, path)


def plot_recommendation_reason_counts(reasons: pd.DataFrame, path: Path, top_n: int = 18) -> Path:
    if reasons is None or reasons.empty or "reason" not in reasons.columns:
        return _empty(path, "No recommendation reasons were available.", "Recommendation reasons")
    df = reasons.copy().head(top_n)
    df["n_features"] = pd.to_numeric(df.get("n_features", 0), errors="coerce").fillna(0)
    fig, ax = plt.subplots(figsize=(10, max(4.8, .38 * len(df))))
    ax.barh(df["reason"].astype(str), df["n_features"], color=GOLD)
    ax.invert_yaxis(); ax.set_xlabel("Number of affected features", color=MUTED)
    _style(ax, "Most common review reasons")
    return _save(fig, path)


def plot_recommendation_family_summary(family: pd.DataFrame, path: Path) -> Path:
    if family is None or family.empty or "family_or_subsystem" not in family.columns:
        return _empty(path, "No family-level recommendation summary was available.", "Recommendations by family")
    df = family.copy().sort_values("median_readiness_score", ascending=True)
    df["median_readiness_score"] = pd.to_numeric(df.get("median_readiness_score", np.nan), errors="coerce")
    fig, ax = plt.subplots(figsize=(10, max(4.8, .42 * len(df))))
    colors = [TEAL if v >= 80 else GOLD if v >= 60 else RED for v in df["median_readiness_score"].fillna(0)]
    ax.barh(df["family_or_subsystem"].astype(str), df["median_readiness_score"], color=colors)
    ax.set_xlim(0, 100); ax.set_xlabel("Median readiness score", color=MUTED)
    ax.axvline(80, color=TEAL, linestyle="--", linewidth=1, alpha=.6)
    ax.axvline(60, color=GOLD, linestyle="--", linewidth=1, alpha=.6)
    ax.axvline(40, color=RED, linestyle="--", linewidth=1, alpha=.6)
    _style(ax, "Feature readiness by family/subsystem")
    return _save(fig, path)


def plot_ml_export_manifest_summary(recs: pd.DataFrame, path: Path) -> Path:
    if recs is None or recs.empty or "ml_export_default" not in recs.columns:
        return _empty(path, "No export manifest was available.", "ML export manifest")
    df = recs.copy()
    included = int(df["ml_export_default"].astype(bool).sum())
    excluded = int(len(df) - included)
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    ax.bar(["Default export", "Hold for review"], [included, excluded], color=[TEAL, GOLD])
    ax.set_ylabel("Number of features", color=MUTED)
    _style(ax, "Transparent ML export manifest")
    for i, v in enumerate([included, excluded]):
        ax.text(i, v + max(1, len(df) * .02), str(v), ha="center", color=NAVY, fontweight="bold")
    ax.text(.5, -.20, "Export labels are review defaults, not final ML feature selection.", transform=ax.transAxes, ha="center", color=MUTED, fontsize=9)
    return _save(fig, path)


def plot_task_counts(df: pd.DataFrame, task_col: str | None, path: Path) -> Path:
    if df is None or df.empty or not task_col or task_col not in df.columns:
        return _empty(path, "No task metadata were available. Load metadata or infer tasks before task review.", "Task coverage")
    s = df[task_col].astype(str).replace({"nan": ""})
    counts = s[s.str.len() > 0].value_counts().sort_values(ascending=True)
    if counts.empty:
        return _empty(path, "Task column was detected, but no usable task values were found.", "Task coverage")
    fig, ax = plt.subplots(figsize=(10, max(4.8, 0.38 * len(counts))))
    ax.barh(counts.index.astype(str), counts.values, color=TEAL)
    ax.set_xlabel("Rows / recordings", color=MUTED)
    _style(ax, "Rows by task")
    for i, v in enumerate(counts.values):
        ax.text(v + max(0.2, counts.max() * 0.01), i, str(int(v)), va="center", fontsize=9, color=MUTED)
    return _save(fig, path)


def plot_task_subject_matrix(df: pd.DataFrame, task_col: str | None, subject_col: str | None, path: Path) -> Path:
    if df is None or df.empty or not task_col or task_col not in df.columns:
        return _empty(path, "No task metadata were available for task x subject coverage.", "Task x subject")
    if not subject_col or subject_col not in df.columns:
        return _empty(path, "No subject_id column was available for task x subject coverage.", "Task x subject")
    mat = pd.crosstab(df[subject_col].astype(str), df[task_col].astype(str))
    mat = mat.loc[mat.index.astype(str) != "", mat.columns.astype(str) != ""]
    if mat.empty:
        return _empty(path, "No usable task x subject matrix could be built.", "Task x subject")
    fig, ax = plt.subplots(figsize=(max(9, 0.34 * len(mat.columns) + 4), max(5, min(16, 0.16 * len(mat.index) + 3))))
    im = ax.imshow((mat > 0).astype(int).to_numpy(), aspect="auto", interpolation="nearest", cmap="viridis", vmin=0, vmax=1)
    ax.set_title("Task x subject coverage", fontsize=14, fontweight="bold", color=NAVY, pad=12)
    ax.set_xlabel(f"Tasks shown: {mat.shape[1]}", color=MUTED)
    ax.set_ylabel(f"Subjects shown: {mat.shape[0]}", color=MUTED)
    ax.set_xticks(range(mat.shape[1]))
    ax.set_xticklabels(mat.columns.astype(str), rotation=45, ha="right", fontsize=8, color=MUTED)
    y_step = max(1, mat.shape[0] // 35)
    yticks = list(range(0, mat.shape[0], y_step))
    ax.set_yticks(yticks)
    ax.set_yticklabels([mat.index[i] for i in yticks], fontsize=7, color=MUTED)
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_ticks([0, 1]); cbar.set_ticklabels(["absent", "present"])
    cbar.ax.tick_params(labelsize=8, colors=MUTED)
    return _save(fig, path)


def plot_task_label_context(df: pd.DataFrame, task_col: str | None, label_col: str | None, path: Path) -> Path:
    if df is None or df.empty or not task_col or task_col not in df.columns:
        return _empty(path, "No task metadata were available for task x label context.", "Task x label")
    if not label_col or label_col not in df.columns:
        return _empty(path, "No diagnosis or severity column was available for task context.", "Task x label")
    task = df[task_col].astype(str).replace({"nan": ""})
    vals = df[label_col]
    numeric = pd.to_numeric(vals, errors="coerce")
    if numeric.notna().sum() >= max(3, len(df) * 0.3):
        g = df.assign(_task=task, _value=numeric).dropna(subset=["_value"])
        if g.empty:
            return _empty(path, "No numeric label/severity values were available.", "Task x severity")
        order = g.groupby("_task")["_value"].median().sort_values().index.tolist()
        data = [g.loc[g["_task"].eq(t), "_value"].dropna().to_numpy() for t in order if str(t)]
        fig, ax = plt.subplots(figsize=(max(9, 0.45 * len(data) + 4), 5.6))
        ax.boxplot(data, labels=[t for t in order if str(t)], vert=True, patch_artist=True)
        ax.set_ylabel(str(label_col), color=MUTED)
        ax.tick_params(axis="x", rotation=45)
        _style(ax, f"{label_col} by task")
        return _save(fig, path)
    tab = pd.crosstab(task, vals.astype(str)).loc[lambda x: x.index.astype(str) != ""]
    if tab.empty:
        return _empty(path, "No categorical label values were available by task.", "Task x diagnosis")
    fig, ax = plt.subplots(figsize=(max(9, 0.45 * tab.shape[1] + 0.3 * tab.shape[0] + 4), max(5, 0.35 * tab.shape[0] + 3)))
    im = ax.imshow(tab.to_numpy(dtype=float), aspect="auto", cmap="Blues")
    ax.set_title(f"{label_col} counts by task", fontsize=14, fontweight="bold", color=NAVY, pad=12)
    ax.set_xticks(range(tab.shape[1])); ax.set_xticklabels(tab.columns.astype(str), rotation=45, ha="right", fontsize=8, color=MUTED)
    ax.set_yticks(range(tab.shape[0])); ax.set_yticklabels(tab.index.astype(str), fontsize=8, color=MUTED)
    for i in range(tab.shape[0]):
        for j in range(tab.shape[1]):
            ax.text(j, i, str(int(tab.iat[i, j])), ha="center", va="center", fontsize=8, color=NAVY)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    return _save(fig, path)


def plot_task_feature_support(task_feature_support: pd.DataFrame, path: Path) -> Path:
    if task_feature_support is None or task_feature_support.empty or "task" not in task_feature_support.columns:
        return _empty(path, "No task feature-support table was available.", "Task feature support")
    df = task_feature_support.copy()
    if "mean_feature_missingness" not in df.columns:
        return _empty(path, "Task feature-support table did not include mean_feature_missingness.", "Task feature support")
    df["mean_feature_missingness"] = pd.to_numeric(df["mean_feature_missingness"], errors="coerce")
    df = df.dropna(subset=["mean_feature_missingness"]).sort_values("mean_feature_missingness", ascending=True)
    if df.empty:
        return _empty(path, "No task-level feature missingness values were available.", "Task feature support")
    fig, ax = plt.subplots(figsize=(10, max(4.8, 0.38 * len(df))))
    colors = [RED if v >= .40 else GOLD if v >= .15 else TEAL for v in df["mean_feature_missingness"]]
    ax.barh(df["task"].astype(str), df["mean_feature_missingness"], color=colors)
    ax.set_xlabel("Mean feature missingness", color=MUTED)
    ax.set_xlim(0, 1)
    _style(ax, "Task-level feature support")
    return _save(fig, path)



def _task_tmp_frame(df: pd.DataFrame, task_col: str | None) -> pd.DataFrame:
    if df is None or df.empty or not task_col or task_col not in df.columns:
        return pd.DataFrame()
    tmp = df.copy()
    tmp["__task__"] = tmp[task_col].astype("string").fillna("").astype(str)
    tmp = tmp[tmp["__task__"].str.strip().ne("")]
    return tmp


def plot_task_readiness_dashboard(
    df: pd.DataFrame,
    task_col: str | None,
    subject_col: str | None,
    clinical_series: pd.Series | None,
    clinical_label: str,
    feature_cols: list[str],
    readiness: pd.DataFrame,
    path: Path,
) -> Path:
    """Clinician-facing task readiness dashboard.

    The goal is not a model; it is a fast orientation screen: completion burden,
    subject breadth, feature completeness, and clinical group coverage by task.
    """
    tmp = _task_tmp_frame(df, task_col)
    if tmp.empty:
        return _empty(path, "No task metadata were available. Load metadata or apply filename task parsing before Task Review.", "Task readiness overview")
    if readiness is None or readiness.empty or "task" not in readiness.columns:
        rows = []
        for task, subdf in tmp.groupby("__task__"):
            rows.append({"task": task, "n_rows": len(subdf), "n_subjects": subdf[subject_col].nunique() if subject_col and subject_col in subdf else 0, "mean_feature_missingness": 0.0})
        readiness = pd.DataFrame(rows)
    r = readiness.copy()
    for col in ["n_rows", "n_subjects", "mean_feature_missingness", "smallest_clinical_group_n"]:
        if col in r.columns:
            r[col] = pd.to_numeric(r[col], errors="coerce")
    r = r.sort_values("n_rows", ascending=False).head(18)
    if r.empty:
        return _empty(path, "No task readiness rows could be built.", "Task readiness overview")

    fig = plt.figure(figsize=(14, max(7.5, 0.42 * len(r) + 4)))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.2], hspace=0.42, wspace=0.30)
    fig.suptitle("Task readiness overview", fontsize=16, fontweight="bold", color=NAVY, y=0.98)

    ax1 = fig.add_subplot(gs[0, 0])
    y = range(len(r))
    ax1.barh(y, r["n_rows"], color=TEAL)
    ax1.set_yticks(y); ax1.set_yticklabels(r["task"].astype(str), fontsize=8, color=MUTED)
    ax1.invert_yaxis(); ax1.set_xlabel("Rows / recordings", color=MUTED)
    _style(ax1, "Completion burden by task")

    ax2 = fig.add_subplot(gs[0, 1])
    subj = r["n_subjects"] if "n_subjects" in r else pd.Series([0] * len(r))
    ax2.barh(y, subj, color=GOLD)
    ax2.set_yticks(y); ax2.set_yticklabels(r["task"].astype(str), fontsize=8, color=MUTED)
    ax2.invert_yaxis(); ax2.set_xlabel("Subjects", color=MUTED)
    _style(ax2, "Subject breadth")

    ax3 = fig.add_subplot(gs[1, 0])
    miss = r["mean_feature_missingness"] if "mean_feature_missingness" in r else pd.Series([np.nan] * len(r))
    colors = [RED if pd.notna(v) and v >= .40 else GOLD if pd.notna(v) and v >= .15 else TEAL for v in miss]
    ax3.barh(y, miss.fillna(0), color=colors)
    ax3.set_yticks(y); ax3.set_yticklabels(r["task"].astype(str), fontsize=8, color=MUTED)
    ax3.invert_yaxis(); ax3.set_xlim(0, 1); ax3.set_xlabel("Mean feature missingness", color=MUTED)
    _style(ax3, "Feature completeness risk")

    ax4 = fig.add_subplot(gs[1, 1])
    if "smallest_clinical_group_n" in r.columns and r["smallest_clinical_group_n"].notna().sum() > 0:
        vals = r["smallest_clinical_group_n"].fillna(0)
        ax4.barh(y, vals, color=TEAL)
        ax4.set_xlabel("Smallest clinical group count", color=MUTED)
        title = f"Clinical balance ({clinical_label})"
    elif clinical_series is not None and len(clinical_series) == len(df):
        aligned = pd.Series(clinical_series.to_numpy(), index=df.index).astype("string").fillna("missing")
        counts = []
        for task in r["task"].astype(str):
            sub = aligned.loc[tmp[tmp["__task__"].eq(task)].index].value_counts()
            counts.append(int(sub.min()) if len(sub) else 0)
        ax4.barh(y, counts, color=TEAL)
        ax4.set_xlabel("Smallest clinical group count", color=MUTED)
        title = f"Clinical balance ({clinical_label})"
    else:
        ax4.text(.5, .5, "No clinical grouping selected", ha="center", va="center", color=MUTED, transform=ax4.transAxes)
        ax4.set_xticks([]); ax4.set_yticks([])
        title = "Clinical balance"
    ax4.set_yticks(y); ax4.set_yticklabels(r["task"].astype(str), fontsize=8, color=MUTED)
    ax4.invert_yaxis(); _style(ax4, title)

    fig.text(0.5, 0.01, "Readiness is descriptive: it flags task support, balance, and completeness before clinical interpretation or modeling.", ha="center", color=MUTED, fontsize=9)
    return _save(fig, path)


def plot_task_clinical_balance_bars(
    df: pd.DataFrame,
    task_col: str | None,
    clinical_series: pd.Series | None,
    clinical_label: str,
    selected_level: str | None,
    path: Path,
) -> Path:
    tmp = _task_tmp_frame(df, task_col)
    if tmp.empty:
        return _empty(path, "No task metadata were available for clinical balance review.", "Task clinical balance")
    if clinical_series is None or len(clinical_series) != len(df):
        return _empty(path, "No diagnosis/severity/context variable was selected or detected.", "Task clinical balance")
    tmp["__clinical__"] = pd.Series(clinical_series.to_numpy(), index=df.index).astype("string").fillna("missing").astype(str)
    tmp = tmp[tmp["__clinical__"].str.strip().ne("")]
    if selected_level and selected_level not in {"All values", "Auto / not available"}:
        counts = tmp[tmp["__clinical__"].eq(str(selected_level))]["__task__"].value_counts().sort_values(ascending=True)
        if counts.empty:
            return _empty(path, f"No task rows for {clinical_label} = {selected_level}.", "Task clinical balance")
        fig, ax = plt.subplots(figsize=(10, max(4.8, 0.38 * len(counts))))
        ax.barh(counts.index.astype(str), counts.values, color=TEAL)
        ax.set_xlabel("Rows / recordings", color=MUTED)
        _style(ax, f"Task representation within {clinical_label} = {selected_level}")
        return _save(fig, path)
    tab = pd.crosstab(tmp["__task__"], tmp["__clinical__"])
    if tab.empty:
        return _empty(path, "No task x clinical grouping counts were available.", "Task clinical balance")
    # Limit visual complexity but keep most represented groups.
    tab = tab.loc[tab.sum(axis=1).sort_values(ascending=False).head(18).index]
    keep_cols = tab.sum(axis=0).sort_values(ascending=False).head(6).index.tolist()
    other = tab.drop(columns=keep_cols).sum(axis=1) if len(tab.columns) > len(keep_cols) else None
    plot_tab = tab[keep_cols].copy()
    if other is not None and other.sum() > 0:
        plot_tab["Other"] = other
    prop = plot_tab.div(plot_tab.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    fig, ax = plt.subplots(figsize=(12, max(5.2, 0.44 * len(prop) + 2)))
    left = np.zeros(len(prop))
    for col in prop.columns:
        vals = prop[col].to_numpy(dtype=float)
        ax.barh(prop.index.astype(str), vals, left=left, label=str(col))
        left += vals
    ax.set_xlim(0, 1)
    ax.set_xlabel("Proportion within task", color=MUTED)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.28), ncol=min(4, len(prop.columns)), frameon=False, fontsize=8)
    _style(ax, f"Clinical balance by task: {clinical_label}")
    fig.text(0.5, 0.02, "Stacked bars show composition, not performance. Check counts in the detailed table before interpreting small groups.", ha="center", color=MUTED, fontsize=9)
    return _save(fig, path)


def plot_task_subject_coverage_summary(df: pd.DataFrame, task_col: str | None, subject_col: str | None, path: Path) -> Path:
    tmp = _task_tmp_frame(df, task_col)
    if tmp.empty:
        return _empty(path, "No task metadata were available for subject coverage.", "Subject-task coverage")
    if not subject_col or subject_col not in tmp.columns:
        return _empty(path, "No subject_id column was available for subject-task coverage.", "Subject-task coverage")
    g = tmp.groupby("__task__").agg(n_rows=(subject_col, "size"), n_subjects=(subject_col, lambda x: x.astype(str).nunique())).reset_index().rename(columns={"__task__":"task"})
    g["records_per_subject"] = g["n_rows"] / g["n_subjects"].replace(0, np.nan)
    g = g.sort_values("n_subjects", ascending=True).tail(18)
    if g.empty:
        return _empty(path, "No usable subject-task coverage rows were available.", "Subject-task coverage")
    fig, ax = plt.subplots(figsize=(11, max(5, 0.42 * len(g) + 2)))
    y = np.arange(len(g))
    ax.barh(y - 0.18, g["n_subjects"], height=0.34, color=TEAL, label="Subjects")
    ax.barh(y + 0.18, g["n_rows"], height=0.34, color=GOLD, label="Rows")
    ax.set_yticks(y); ax.set_yticklabels(g["task"].astype(str), fontsize=8, color=MUTED)
    ax.set_xlabel("Count", color=MUTED)
    ax.legend(frameon=False, loc="lower right")
    _style(ax, "Subject-task coverage")
    fig.text(0.5, 0.02, "Tasks with many rows but few subjects may reflect repeated recordings rather than broad cohort coverage.", ha="center", color=MUTED, fontsize=9)
    return _save(fig, path)


def plot_task_feature_profile(df: pd.DataFrame, task_col: str | None, feature_cols: list[str], selected_task: str | None, path: Path) -> Path:
    tmp = _task_tmp_frame(df, task_col)
    if tmp.empty:
        return _empty(path, "No task metadata were available for task-specific feature profiles.", "Task-specific feature profile")
    feature_cols = [c for c in (feature_cols or []) if c in tmp.columns and pd.api.types.is_numeric_dtype(tmp[c])]
    if not feature_cols:
        return _empty(path, "No mapped numeric feature columns were available.", "Task-specific feature profile")
    selected_task = str(selected_task or "")
    all_tasks = {"", "All tasks", "All tasks / not available"}
    rows = []
    if selected_task not in all_tasks and selected_task in set(tmp["__task__"].astype(str)):
        in_task = tmp["__task__"].astype(str).eq(selected_task)
        if in_task.sum() < 3 or (~in_task).sum() < 3:
            return _empty(path, "Selected task has too few rows for a feature profile against the remaining data.", "Task-specific feature profile")
        for c in feature_cols:
            x = pd.to_numeric(tmp.loc[in_task, c], errors="coerce").dropna()
            y = pd.to_numeric(tmp.loc[~in_task, c], errors="coerce").dropna()
            if len(x) < 3 or len(y) < 3:
                continue
            pooled = pd.concat([x, y])
            spread = float(pooled.std(ddof=0))
            if not np.isfinite(spread) or spread == 0:
                continue
            z = float((x.median() - y.median()) / spread)
            rows.append({"feature": c, "effect": z, "abs_effect": abs(z), "task_median": float(x.median()), "other_median": float(y.median())})
        title = f"Feature profile for {selected_task}"
        xlabel = "Median difference vs other tasks (SD units)"
    else:
        for c in feature_cols:
            vals = pd.to_numeric(tmp[c], errors="coerce")
            if vals.notna().sum() < 6 or vals.std(ddof=0) == 0:
                continue
            med = tmp.assign(__v__=vals).groupby("__task__")["__v__"].median().dropna()
            if len(med) < 2:
                continue
            effect = float((med.max() - med.min()) / vals.std(ddof=0))
            rows.append({"feature": c, "effect": effect, "abs_effect": abs(effect), "task_median": float(med.max()), "other_median": float(med.min())})
        title = "Features most separated across tasks"
        xlabel = "Between-task median spread (SD units)"
    out = pd.DataFrame(rows).sort_values("abs_effect", ascending=True).tail(18) if rows else pd.DataFrame()
    if out.empty:
        return _empty(path, "No stable task-feature contrasts could be computed.", "Task-specific feature profile")
    fig, ax = plt.subplots(figsize=(11, max(5, 0.38 * len(out) + 2)))
    vals = out["effect"].to_numpy(dtype=float)
    colors = [TEAL if v >= 0 else GOLD for v in vals]
    ax.barh(out["feature"].astype(str), vals, color=colors)
    ax.axvline(0, color=NAVY, lw=1)
    ax.set_xlabel(xlabel, color=MUTED)
    _style(ax, title)
    fig.text(0.5, 0.02, "This is descriptive and task-contextual. It identifies task-sensitive features; it is not a clinical outcome test.", ha="center", color=MUTED, fontsize=9)
    return _save(fig, path)

def plot_longitudinal_subject_records(df: pd.DataFrame, subject_col: str | None, path: Path) -> Path:
    if df is None or df.empty or not subject_col or subject_col not in df.columns:
        return _empty(path, "No subject_id column was available for repeated-subject review.", "Subject repeats")
    counts = df[subject_col].astype(str).replace({"nan": ""})
    counts = counts[counts.str.len() > 0].value_counts().sort_values(ascending=True)
    if counts.empty:
        return _empty(path, "No usable subject IDs were found.", "Subject repeats")
    fig, ax = plt.subplots(figsize=(10, max(5, min(16, 0.18 * len(counts) + 3))))
    colors = [TEAL if v > 1 else MUTED for v in counts.values]
    ax.barh(counts.index.astype(str), counts.values, color=colors)
    ax.set_xlabel("Records / rows", color=MUTED)
    _style(ax, "Records per subject")
    y_step = max(1, len(counts) // 45)
    for label in ax.get_yticklabels():
        label.set_visible(False)
    ax.set_yticks(range(0, len(counts), y_step))
    ax.set_yticklabels([counts.index[i] for i in range(0, len(counts), y_step)], fontsize=7, color=MUTED)
    return _save(fig, path)


def plot_longitudinal_readiness(readiness: pd.DataFrame, path: Path) -> Path:
    """Repeated-record cohort summary for longitudinal review.

    This first longitudinal plot intentionally avoids plotting every subject.
    It filters to subject-task units with at least two recordings of the same
    task, then summarizes cohort depth, dated follow-up span, and task-level
    repeated-record coverage with robust statistics.
    """
    if readiness is None or readiness.empty or "status" in readiness.columns:
        return _empty(path, "No longitudinal readiness table could be built.", "Repeated-record cohort summary")
    d = readiness.copy()
    if "n_records" not in d.columns:
        return _empty(path, "Readiness table is missing n_records.", "Repeated-record cohort summary")
    for col in ["n_records", "n_sessions", "n_iterations", "n_dates", "elapsed_days"]:
        if col in d.columns:
            d[col] = pd.to_numeric(d[col], errors="coerce")
        else:
            d[col] = pd.NA
    if "subject_id" not in d.columns:
        d["subject_id"] = "subject"
    if "task" not in d.columns:
        d["task"] = "all_tasks"

    repeated = d.loc[d["n_records"].fillna(0) >= 2].copy()
    if repeated.empty:
        total_subjects = int(d["subject_id"].astype(str).nunique()) if "subject_id" in d.columns else 0
        fig, ax = plt.subplots(figsize=(10.5, 5.8))
        ax.axis("off")
        ax.text(0.5, 0.62, "No repeated same-task recordings found", ha="center", va="center", fontsize=16, fontweight="bold", color=NAVY)
        ax.text(0.5, 0.48, f"Evaluated {len(d)} subject-task units across {total_subjects} subjects. Longitudinal feature-change plots require at least two recordings of the same task for a subject.", ha="center", va="center", fontsize=11, color=MUTED, wrap=True)
        return _save(fig, path)

    total_subjects = int(d["subject_id"].astype(str).nunique())
    repeated_subjects = int(repeated["subject_id"].astype(str).nunique())
    repeated_units = int(len(repeated))
    repeated_records = int(repeated["n_records"].fillna(0).sum())
    dated = repeated.loc[repeated["elapsed_days"].notna()].copy()
    dated_units = int(len(dated))
    med_records = float(repeated["n_records"].median())
    mean_records = float(repeated["n_records"].mean())
    med_days = float(dated["elapsed_days"].median()) if dated_units else np.nan
    mean_days = float(dated["elapsed_days"].mean()) if dated_units else np.nan

    fig, axes = plt.subplots(2, 2, figsize=(14.2, 9.2), gridspec_kw={"height_ratios": [1.0, 1.15]})
    ax0, ax1, ax2, ax3 = axes.ravel()

    # Panel 1: cohort eligibility counts.
    counts = pd.DataFrame({
        "metric": ["All subjects", "Repeated subjects", "Repeated subject-task units", "Repeated recordings"],
        "count": [total_subjects, repeated_subjects, repeated_units, repeated_records],
    })
    ax0.barh(counts["metric"], counts["count"], color=[MUTED, TEAL, TEAL, GOLD])
    ax0.invert_yaxis()
    ax0.set_xlabel("Count", color=MUTED)
    _style(ax0, "Repeated-record cohort")
    ax0.text(0.02, -0.22, "Inclusion rule for later plots: subject + same task + at least 2 recordings.", transform=ax0.transAxes, fontsize=9, color=MUTED, va="top")

    # Panel 2: recording depth distribution.
    rec = repeated["n_records"].dropna().astype(int)
    bins = [2, 3, 4, 5, 6, 999999]
    labels = ["2", "3", "4", "5", "6+"]
    cats = pd.cut(rec, bins=[1, 2, 3, 4, 5, 999999], labels=labels, right=True)
    depth = cats.value_counts().reindex(labels, fill_value=0)
    ax1.bar(depth.index.astype(str), depth.values, color=TEAL)
    ax1.set_xlabel("Recordings per subject-task unit", color=MUTED)
    ax1.set_ylabel("Number of units", color=MUTED)
    _style(ax1, "Recording depth")
    ax1.text(0.02, 0.95, f"Median: {med_records:.1f}\nMean: {mean_records:.1f}", transform=ax1.transAxes, ha="left", va="top", fontsize=9, color=NAVY, bbox=dict(facecolor="white", edgecolor=GRID, boxstyle="round,pad=0.35"))

    # Panel 3: elapsed-day distribution for dated repeated units.
    if dated_units:
        spans = dated["elapsed_days"].dropna().astype(float)
        max_span = max(float(spans.max()), 1.0)
        edges = [0, 30, 90, 180, 365, 730]
        if max_span > 730:
            edges.append(max_span + 1)
            span_labels = ["0–30", "31–90", "91–180", "181–365", "366–730", ">730"]
        else:
            edges.append(731)
            span_labels = ["0–30", "31–90", "91–180", "181–365", "366–730", ">730"]
        span_cat = pd.cut(spans, bins=edges, labels=span_labels[:len(edges)-1], include_lowest=True, right=True)
        span_counts = span_cat.value_counts().reindex(span_labels[:len(edges)-1], fill_value=0)
        ax2.bar(span_counts.index.astype(str), span_counts.values, color=GOLD)
        ax2.set_xlabel("Days between first and last dated recording", color=MUTED)
        ax2.set_ylabel("Number of units", color=MUTED)
        _style(ax2, "Follow-up span distribution")
        ax2.text(0.02, 0.95, f"Dated units: {dated_units}/{repeated_units}\nMedian: {med_days:.0f} days\nMean: {mean_days:.0f} days", transform=ax2.transAxes, ha="left", va="top", fontsize=9, color=NAVY, bbox=dict(facecolor="white", edgecolor=GRID, boxstyle="round,pad=0.35"))
    else:
        ax2.axis("off")
        ax2.text(0.5, 0.58, "No dated follow-up spans", ha="center", va="center", fontsize=13, fontweight="bold", color=NAVY)
        ax2.text(0.5, 0.42, "Repeated recordings exist, but date fields are unavailable or not parseable. Later change plots can use visit/session/iteration order, but not days between visits.", ha="center", va="center", fontsize=10, color=MUTED, wrap=True)

    # Panel 4: task-level repeated coverage, compact top-N summary.
    task = repeated["task"].astype(str).replace({"nan": "Unknown", "": "Unknown"})
    task_summary = repeated.assign(__task=task).groupby("__task", dropna=False).agg(
        repeated_subjects=("subject_id", lambda x: int(pd.Series(x).astype(str).nunique())),
        repeated_units=("n_records", "size"),
        median_records=("n_records", "median"),
        median_elapsed_days=("elapsed_days", "median"),
    ).reset_index().sort_values(["repeated_subjects", "repeated_units"], ascending=True).tail(12)
    if task_summary.empty:
        ax3.axis("off")
        ax3.text(0.5, 0.5, "No task-level repeated coverage available", ha="center", va="center", color=MUTED)
    else:
        labels = task_summary["__task"].astype(str)
        ax3.barh(labels, task_summary["repeated_subjects"].astype(float), color=TEAL)
        ax3.set_xlabel("Subjects with repeated same-task recordings", color=MUTED)
        _style(ax3, "Repeated coverage by task")
        xmax = max(float(task_summary["repeated_subjects"].max()), 1.0)
        for y, (_, row) in enumerate(task_summary.iterrows()):
            med_rec = row.get("median_records", np.nan)
            med_span = row.get("median_elapsed_days", np.nan)
            span_txt = "no dates" if pd.isna(med_span) else f"median {med_span:.0f} d"
            ax3.text(float(row["repeated_subjects"]) + xmax * 0.02, y, f"units={int(row['repeated_units'])}, records={med_rec:.1f}, {span_txt}", va="center", fontsize=8, color=MUTED)
        ax3.set_xlim(0, xmax * 1.55)

    fig.suptitle("Longitudinal repeated-record cohort summary", fontsize=15, fontweight="bold", color=NAVY, y=0.985)
    fig.text(0.5, 0.018, "Interpretation: later longitudinal feature-change plots should be restricted to repeated subject-task units. This screen quantifies how large and how temporally deep that analyzable cohort is.", ha="center", color=MUTED, fontsize=9)
    return _save(fig, path)

def plot_longitudinal_visit_timeline(visits: pd.DataFrame, path: Path) -> Path:
    """Selected subject-task visit timeline for longitudinal review.

    This plot is intentionally subject-level and only becomes active after the
    user selects one repeated subject in the Longitudinal menu. It verifies
    visit ordering and temporal spacing before feature-change plots are added.
    """
    title = "Selected subject-task visit timeline"
    if visits is None or visits.empty or "status" in visits.columns:
        msg = "Select one repeated subject to view their visit timeline."
        if isinstance(visits, pd.DataFrame) and not visits.empty and "status" in visits.columns:
            msg = str(visits["status"].iloc[0])
        return _empty(path, msg, title)
    d = visits.copy()
    if "record_order" not in d.columns:
        d["record_order"] = np.arange(1, len(d) + 1)
    order_values = pd.to_numeric(d["record_order"], errors="coerce")
    fallback_order = pd.Series(np.arange(1, len(d) + 1), index=d.index)
    d["record_order"] = order_values.where(order_values.notna(), fallback_order).astype(int)
    if "days_since_first" in d.columns:
        d["days_since_first"] = pd.to_numeric(d["days_since_first"], errors="coerce")
    else:
        d["days_since_first"] = np.nan
    if d["days_since_first"].notna().sum() >= 2:
        x = d["days_since_first"].astype(float)
        xlabel = "Days since first dated recording"
    else:
        x = d["record_order"].astype(float)
        xlabel = "Recording order (dates unavailable or incomplete)"
    subject = str(d["subject_id"].dropna().iloc[0]) if "subject_id" in d.columns and d["subject_id"].notna().any() else "selected subject"
    task = str(d["task"].dropna().iloc[0]) if "task" in d.columns and d["task"].notna().any() else "selected task"
    fig, ax = plt.subplots(figsize=(12.5, 4.8))
    y = np.zeros(len(d))
    ax.plot(x, y, color=GRID, linewidth=3, zorder=1)
    ax.scatter(x, y, s=95, color=TEAL, edgecolor="white", linewidth=1.5, zorder=3)
    for _, row in d.iterrows():
        xv = float(row["days_since_first"]) if pd.notna(row.get("days_since_first", np.nan)) and d["days_since_first"].notna().sum() >= 2 else float(row["record_order"])
        order = int(row.get("record_order", 0))
        date_txt = str(row.get("date", "")).strip()
        interval = row.get("interval_from_previous_days", np.nan)
        session = str(row.get("session_or_visit", "")).strip()
        iteration = str(row.get("iteration", "")).strip()
        label_lines = [f"#{order}"]
        if date_txt:
            label_lines.append(date_txt)
        if pd.notna(interval):
            label_lines.append(f"+{int(interval)} d")
        elif order == 1:
            label_lines.append("baseline")
        context = ", ".join([v for v in [f"visit {session}" if session else "", f"iter {iteration}" if iteration else ""] if v])
        if context:
            label_lines.append(context[:28])
        va = "bottom" if order % 2 else "top"
        dy = 0.08 if order % 2 else -0.08
        ax.text(xv, dy, "\n".join(label_lines), ha="center", va=va, fontsize=8, color=NAVY)
    ax.set_yticks([])
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_ylim(-0.42, 0.42)
    xmin, xmax = float(np.nanmin(x)), float(np.nanmax(x))
    pad = max((xmax - xmin) * 0.08, 1.0)
    ax.set_xlim(xmin - pad, xmax + pad)
    _style(ax, title)
    ax.grid(axis="x", color=GRID, alpha=0.75, linewidth=0.8)
    total_span = xmax - xmin if len(d) > 1 else 0
    fig.suptitle(f"{subject} | {task}", fontsize=12, fontweight="bold", color=NAVY, y=0.98)
    fig.text(0.5, 0.03, f"{len(d)} recordings shown. Total displayed span: {total_span:.0f} {'days' if xlabel.startswith('Days') else 'recording-order units'}. Verify this timeline before interpreting feature trajectories.", ha="center", color=MUTED, fontsize=9)
    return _save(fig, path)

def plot_longitudinal_session_matrix(df: pd.DataFrame, subject_col: str | None, session_col: str | None, path: Path) -> Path:
    if df is None or df.empty or not subject_col or subject_col not in df.columns:
        return _empty(path, "No subject_id column was available for session/visit review.", "Session / visit")
    if not session_col or session_col not in df.columns:
        return _empty(path, "No session_id or visit_id column was available.", "Session / visit")
    mat = pd.crosstab(df[subject_col].astype(str), df[session_col].astype(str))
    mat = mat.loc[mat.index.astype(str) != "", mat.columns.astype(str) != ""]
    if mat.empty:
        return _empty(path, "No usable subject x session matrix could be built.", "Session / visit")
    fig, ax = plt.subplots(figsize=(max(9, 0.34 * mat.shape[1] + 4), max(5, min(16, 0.16 * mat.shape[0] + 3))))
    im = ax.imshow((mat > 0).astype(int).to_numpy(), aspect="auto", interpolation="nearest", cmap="viridis", vmin=0, vmax=1)
    ax.set_title("Subject x session/visit coverage", fontsize=14, fontweight="bold", color=NAVY, pad=12)
    ax.set_xlabel("Session / visit", color=MUTED)
    ax.set_ylabel("Subject", color=MUTED)
    ax.set_xticks(range(mat.shape[1])); ax.set_xticklabels(mat.columns.astype(str), rotation=45, ha="right", fontsize=8, color=MUTED)
    y_step = max(1, mat.shape[0] // 35)
    ax.set_yticks(range(0, mat.shape[0], y_step)); ax.set_yticklabels([mat.index[i] for i in range(0, mat.shape[0], y_step)], fontsize=7, color=MUTED)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    return _save(fig, path)


def plot_longitudinal_iteration_counts(df: pd.DataFrame, subject_col: str | None, iteration_col: str | None, path: Path) -> Path:
    if df is None or df.empty or not subject_col or subject_col not in df.columns:
        return _empty(path, "No subject_id column was available for iteration review.", "Iterations")
    if not iteration_col or iteration_col not in df.columns:
        return _empty(path, "No iteration column was available.", "Iterations")
    tab = pd.crosstab(df[subject_col].astype(str), df[iteration_col].astype(str))
    tab = tab.loc[tab.index.astype(str) != "", tab.columns.astype(str) != ""]
    if tab.empty:
        return _empty(path, "No usable subject x iteration table could be built.", "Iterations")
    fig, ax = plt.subplots(figsize=(max(9, 0.4 * tab.shape[1] + 4), max(5, min(16, 0.16 * tab.shape[0] + 3))))
    im = ax.imshow(tab.to_numpy(dtype=float), aspect="auto", interpolation="nearest", cmap="Blues")
    ax.set_title("Subject x iteration counts", fontsize=14, fontweight="bold", color=NAVY, pad=12)
    ax.set_xlabel("Iteration", color=MUTED); ax.set_ylabel("Subject", color=MUTED)
    ax.set_xticks(range(tab.shape[1])); ax.set_xticklabels(tab.columns.astype(str), rotation=45, ha="right", fontsize=8, color=MUTED)
    y_step = max(1, tab.shape[0] // 35)
    ax.set_yticks(range(0, tab.shape[0], y_step)); ax.set_yticklabels([tab.index[i] for i in range(0, tab.shape[0], y_step)], fontsize=7, color=MUTED)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    return _save(fig, path)


def plot_longitudinal_date_timeline(df: pd.DataFrame, subject_col: str | None, date_col: str | None, path: Path) -> Path:
    if df is None or df.empty or not subject_col or subject_col not in df.columns:
        return _empty(path, "No subject_id column was available for visit-date review.", "Visit dates")
    if not date_col or date_col not in df.columns:
        return _empty(path, "No recording_date or visit_date column was available.", "Visit dates")
    d = df[[subject_col, date_col]].copy()
    d["_date"] = pd.to_datetime(d[date_col], errors="coerce")
    d = d.dropna(subset=["_date"])
    if d.empty:
        return _empty(path, "Visit/date field was detected, but no parseable dates were found.", "Visit dates")
    subjects = sorted(d[subject_col].astype(str).unique().tolist())
    ymap = {s: i for i, s in enumerate(subjects)}
    fig, ax = plt.subplots(figsize=(11, max(5, min(16, 0.18 * len(subjects) + 3))))
    ax.scatter(d["_date"], d[subject_col].astype(str).map(ymap), s=34, color=TEAL, alpha=0.85)
    ax.set_title("Visit / recording dates by subject", fontsize=14, fontweight="bold", color=NAVY, pad=12)
    ax.set_xlabel("Date", color=MUTED); ax.set_ylabel("Subject", color=MUTED)
    y_step = max(1, len(subjects) // 35)
    ax.set_yticks(range(0, len(subjects), y_step)); ax.set_yticklabels([subjects[i] for i in range(0, len(subjects), y_step)], fontsize=7, color=MUTED)
    _style(ax)
    return _save(fig, path)


def _empty_context_plot(path: Path, title: str, message: str) -> Path:
    return _empty(path, message, title)


def plot_task_clinical_context(
    df: pd.DataFrame,
    task_col: str | None,
    clinical_series: pd.Series | None,
    clinical_label: str,
    path: Path,
) -> Path:
    """Task x clinical-context matrix for local Task Review plotting.

    clinical_series is computed locally by the GUI and is not written back into
    the analysis dataframe. This keeps metadata optional and non-disruptive.
    """
    if df is None or df.empty:
        return _empty_context_plot(path, "Task x clinical context", "No analysis table is available.")
    if not task_col or task_col not in df.columns:
        return _empty_context_plot(path, "Task x clinical context", "No task column was detected.")
    if clinical_series is None or len(clinical_series) != len(df):
        return _empty_context_plot(path, "Task x clinical context", "No clinical grouping variable was selected or detected.")
    tmp = pd.DataFrame({
        "task": df[task_col].astype(str).replace({"nan": ""}),
        "clinical_group": clinical_series.astype("string").fillna("missing").astype(str),
    })
    tmp = tmp[(tmp["task"].str.len() > 0) & (tmp["clinical_group"].str.len() > 0)]
    if tmp.empty:
        return _empty_context_plot(path, "Task x clinical context", "Task and clinical grouping values were not usable.")
    tab = pd.crosstab(tmp["task"], tmp["clinical_group"])
    if tab.empty:
        return _empty_context_plot(path, "Task x clinical context", "No task x clinical context counts were available.")
    fig, ax = plt.subplots(figsize=(max(9, 0.55 * tab.shape[1] + 0.35 * tab.shape[0] + 5), max(5, 0.35 * tab.shape[0] + 3)))
    im = ax.imshow(tab.to_numpy(dtype=float), aspect="auto", cmap="Blues")
    ax.set_title(f"Task x {clinical_label}", fontsize=14, fontweight="bold", color=NAVY, pad=12)
    ax.set_xlabel(clinical_label, color=MUTED)
    ax.set_ylabel("Task", color=MUTED)
    ax.set_xticks(range(tab.shape[1]))
    ax.set_xticklabels(tab.columns.astype(str), rotation=45, ha="right", fontsize=8, color=MUTED)
    ax.set_yticks(range(tab.shape[0]))
    ax.set_yticklabels(tab.index.astype(str), fontsize=8, color=MUTED)
    for i in range(tab.shape[0]):
        for j in range(tab.shape[1]):
            ax.text(j, i, str(int(tab.iat[i, j])), ha="center", va="center", fontsize=8, color=NAVY)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    return _save(fig, path)


def plot_task_clinical_filtered_counts(
    df: pd.DataFrame,
    task_col: str | None,
    clinical_series: pd.Series | None,
    clinical_label: str,
    selected_level: str | None,
    path: Path,
) -> Path:
    if df is None or df.empty:
        return _empty_context_plot(path, "Task filtered by clinical group", "No analysis table is available.")
    if not task_col or task_col not in df.columns:
        return _empty_context_plot(path, "Task filtered by clinical group", "No task column was detected.")
    if clinical_series is None or len(clinical_series) != len(df):
        return _empty_context_plot(path, "Task filtered by clinical group", "No clinical grouping variable was selected or detected.")
    tmp = pd.DataFrame({
        "task": df[task_col].astype(str).replace({"nan": ""}),
        "clinical_group": clinical_series.astype("string").fillna("missing").astype(str),
    })
    if selected_level and selected_level not in {"All values", "Auto / not available"}:
        tmp = tmp[tmp["clinical_group"].eq(str(selected_level))]
    tmp = tmp[tmp["task"].str.len() > 0]
    if tmp.empty:
        return _empty_context_plot(path, "Task filtered by clinical group", "The selected clinical level contains no task rows.")
    counts = tmp["task"].value_counts().sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(10, max(4.8, 0.38 * len(counts))))
    ax.barh(counts.index.astype(str), counts.values, color=TEAL)
    ax.set_xlabel("Rows / recordings", color=MUTED)
    title_suffix = f"{clinical_label} = {selected_level}" if selected_level and selected_level != "All values" else f"all {clinical_label}"
    _style(ax, f"Task counts within {title_suffix}")
    for i, v in enumerate(counts.values):
        ax.text(v + max(0.2, counts.max() * 0.01), i, str(int(v)), va="center", fontsize=9, color=MUTED)
    return _save(fig, path)


def plot_qc_framework_selection(framework_table: pd.DataFrame, selected_framework: str, path: Path) -> Path:
    """Readable QC-framework selector plot.

    This plot explains which QC metric family is being used for the current local
    QC scope. It is intentionally descriptive and works when no QC table exists.
    """
    if framework_table is None or framework_table.empty:
        return _empty(path, "No QC framework information was available.", "QC framework")
    ft = framework_table.copy()
    if "framework" not in ft.columns:
        return _empty(path, "QC framework table did not include a framework column.", "QC framework")
    ft["selected"] = ft["framework"].astype(str).eq(str(selected_framework))
    if "n_numeric_qc" not in ft.columns:
        ft["n_numeric_qc"] = 0
    ft["n_numeric_qc"] = pd.to_numeric(ft["n_numeric_qc"], errors="coerce").fillna(0)
    fig, ax = plt.subplots(figsize=(10.5, max(4.5, 1.0 + 0.8 * len(ft))))
    colors = [TEAL if x else "#CFDAE6" for x in ft["selected"]]
    ax.barh(ft["framework"].astype(str), ft["n_numeric_qc"], color=colors)
    ax.invert_yaxis()
    ax.set_xlabel("Detected numeric QC metrics", color=MUTED)
    _style(ax, f"QC framework selection: {selected_framework}")
    for i, row in ft.reset_index(drop=True).iterrows():
        ax.text(float(row["n_numeric_qc"]) + max(0.1, float(ft["n_numeric_qc"].max()) * 0.02 + 0.1), i,
                str(int(row["n_numeric_qc"])), va="center", fontsize=9, color=MUTED)
    footer = "Auto uses all numeric QC metrics. Acoustic/Kinematic modes use conservative column-name heuristics and keep identifiers for alignment."
    ax.text(0, -0.85, footer, fontsize=9, color=MUTED)
    return _save(fig, path)


def plot_overview_design_from_tables(metrics: pd.DataFrame, counts: pd.DataFrame, path: Path) -> Path:
    """Plot Overview design context from pre-aggregated tables only.

    This avoids row/feature dimension coupling inside the plotting layer. The
    GUI computes metadata-first counts before calling this function.
    """
    if metrics is None or metrics.empty:
        return _empty(path, "No Overview design metrics were available.", "Dataset design context")
    m = metrics.copy()
    if "metric" not in m.columns or "value" not in m.columns:
        return _empty(path, "Overview design metrics are missing required columns.", "Dataset design context")
    m["metric"] = m["metric"].astype(str)
    m["value"] = pd.to_numeric(m["value"], errors="coerce").fillna(0)
    c = counts.copy() if counts is not None else pd.DataFrame(columns=["section", "level", "count"])
    for col in ["section", "level"]:
        if col not in c.columns:
            c[col] = ""
    if "count" not in c.columns:
        c["count"] = 0
    c["count"] = pd.to_numeric(c["count"], errors="coerce").fillna(0)

    max_labels = int(c.groupby("section").size().max()) if not c.empty else 1
    fig_h = max(10.2, 8.2 + 0.18 * min(max_labels, 24))
    fig = plt.figure(figsize=(15.8, fig_h))
    gs = fig.add_gridspec(3, 2, height_ratios=[0.9, 1.35, 1.55], hspace=0.58, wspace=0.36)
    ax0 = fig.add_subplot(gs[0, :]); ax0.axis("off")
    ax0.text(0.00, 0.82, "Dataset design context", fontsize=18, fontweight="bold", color=NAVY, va="center")
    ax0.text(0.00, 0.50, "Metadata-first counts from accepted Metadata Mapping; filename context is used only when metadata is absent.", fontsize=10.5, color=MUTED, va="center")
    card_metrics = ["subjects", "tasks", "rows/files", "subjects eligible for repeated analysis"]
    for i, metric in enumerate(card_metrics):
        row = m.loc[m["metric"].eq(metric)]
        val = int(row["value"].iloc[0]) if not row.empty else 0
        x = 0.02 + i * 0.24
        ax0.add_patch(plt.Rectangle((x, 0.04), 0.20, 0.30, transform=ax0.transAxes, color="#F7FAFD", ec=GRID, lw=0.8))
        ax0.text(x + 0.015, 0.25, metric.replace("subjects eligible for repeated analysis", "Repeated-analysis subjects").title(), transform=ax0.transAxes, fontsize=9, color=MUTED, fontweight="bold", va="center")
        ax0.text(x + 0.015, 0.11, str(val), transform=ax0.transAxes, fontsize=18, color=NAVY, fontweight="bold", va="center")

    sections = [
        ("sex_or_gender", "Sex / gender counts", "subjects/rows"),
        ("diagnosis", "Diagnosis counts", "subjects/rows"),
        ("task", "Task counts", "records"),
        ("sessions_per_subject", "Sessions per subject", "subjects"),
    ]
    axes = [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1]), fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[2, 1])]
    for ax, (section, title, xlabel) in zip(axes, sections):
        sub = c.loc[c["section"].astype(str).eq(section)].copy()
        sub = sub.sort_values("count", ascending=False).head(15).sort_values("count", ascending=True)
        if sub.empty:
            ax.text(0.5, 0.5, "not available", ha="center", va="center", color=MUTED)
            ax.set_title(title, fontsize=12, fontweight="bold", color=NAVY)
            ax.axis("off")
            continue
        labels = [str(x) for x in sub["level"]]
        vals = sub["count"].astype(float).to_numpy()
        ax.barh(labels, vals, color=TEAL)
        for j, v in enumerate(vals):
            ax.text(v + max(vals) * 0.015, j, f"{int(v)}", va="center", fontsize=8, color=MUTED)
        ax.set_xlabel(xlabel, color=MUTED)
        ax.tick_params(axis="y", labelsize=8.2)
        _style(ax, title)
    fig.subplots_adjust(left=0.17, right=0.98)
    return _save(fig, path)


def plot_subject_task_summary_bars(summary: pd.DataFrame, path: Path) -> Path:
    """Render subject/task coverage from a pre-aggregated table."""
    required = {"task", "records", "subjects", "repeated_subjects"}
    if summary is None or summary.empty or not required.issubset(summary.columns):
        return _empty(path, "No pre-aggregated subject/task coverage table was available.", "Subject × task coverage")
    df = summary.copy()
    for c in ["records", "subjects", "repeated_subjects"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    df = df.sort_values(["subjects", "records"], ascending=True).tail(24)
    fig, ax = plt.subplots(figsize=(12.4, max(5.5, 0.42 * len(df))))
    y = np.arange(len(df))
    width = 0.25
    ax.barh(y - width, df["records"], height=width, color=TEAL, label="records")
    ax.barh(y, df["subjects"], height=width, color=GOLD, label="subjects")
    ax.barh(y + width, df["repeated_subjects"], height=width, color=NAVY, label="subjects with ≥2 iterations/records")
    ax.set_yticks(y); ax.set_yticklabels([str(x) for x in df["task"]], color=MUTED, fontsize=8.5)
    ax.set_xlabel("Count", color=MUTED)
    ax.legend(frameon=False, fontsize=8)
    _style(ax, "Subject × task coverage counts")
    fig.subplots_adjust(left=0.26, right=0.98)
    return _save(fig, path)
