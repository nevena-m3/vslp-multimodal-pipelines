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


def plot_feature_availability_heatmap(df: pd.DataFrame, feature_cols: Sequence[str], path: Path, max_features: int = 80, max_rows: int = 300) -> Path:
    cols = list(feature_cols)[:max_features]
    if df is None or df.empty or not cols:
        return _empty(path, "No feature columns were detected for the availability matrix.", "Feature availability")
    show = df.loc[:, cols].head(max_rows)
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


def plot_comissing_heatmap(feature_df: pd.DataFrame, feature_cols: Sequence[str], path: Path, max_features: int = 45) -> Path:
    cols = [c for c in list(feature_cols) if c in feature_df.columns]
    if not cols:
        return _empty(path, "No feature columns were available for co-missingness analysis.", "Co-missingness")
    miss_fr = feature_df[cols].isna().mean().sort_values(ascending=False)
    cols = miss_fr.head(max_features).index.tolist()
    if len(cols) < 2:
        return _empty(path, "At least two feature columns are required for co-missingness analysis.", "Co-missingness")
    miss = feature_df[cols].isna().astype(float)
    mat = miss.T.dot(miss) / max(1, len(miss))
    fig, ax = plt.subplots(figsize=(max(8, 0.24 * len(cols)), max(7, 0.24 * len(cols))))
    im = ax.imshow(mat.to_numpy(), vmin=0, vmax=min(1, max(0.01, float(np.nanmax(mat.to_numpy())))), cmap="magma", aspect="auto")
    ax.set_title("Pairwise co-missingness among most-missing features", fontsize=14, fontweight="bold", color=NAVY, pad=12)
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
    required = {"variable_type", "column", "n_unique", "n_missing"}
    if design is None or design.empty or not required.issubset(design.columns):
        return _empty(path, "No dataset-design summary was available.", "Dataset design")
    df = design.copy().head(12)
    n = len(df)
    ncols = 2
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, max(4.8, 2.0 * nrows)))
    axes = np.array(axes).reshape(-1)
    for ax, (_, r) in zip(axes, df.iterrows()):
        col = str(r.get("column", "not_detected"))
        detected = col != "not_detected"
        n_unique = int(pd.to_numeric(pd.Series([r.get("n_unique", 0)]), errors="coerce").fillna(0).iloc[0])
        n_missing = int(pd.to_numeric(pd.Series([r.get("n_missing", 0)]), errors="coerce").fillna(0).iloc[0])
        status = "detected" if detected else "missing"
        ax.set_facecolor("#F8FBFE" if detected else "#FFF7F5")
        ax.text(0.04, 0.76, str(r.get("variable_type", "variable")).replace("_", " ").title(), transform=ax.transAxes, fontsize=12, fontweight="bold", color=NAVY)
        ax.text(0.04, 0.52, col, transform=ax.transAxes, fontsize=10, color=TEAL if detected else RED)
        ax.text(0.04, 0.30, f"{status} · unique={n_unique} · missing={n_missing}", transform=ax.transAxes, fontsize=9, color=MUTED)
        top = str(r.get("top_values", ""))
        if top and top != "nan":
            ax.text(0.04, 0.10, top[:120] + ("…" if len(top) > 120 else ""), transform=ax.transAxes, fontsize=8, color=MUTED)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID)
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle("Detected dataset design variables", fontsize=15, fontweight="bold", color=NAVY, y=1.01)
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
    if df is None or df.empty:
        return _empty(path, "No feature table was available.", "Subject × task coverage")
    subject_col = next((c for c in ["subject_id", "participant_id", "patient_id"] if c in df.columns), None)
    task_col = next((c for c in ["task", "task_name", "prompt"] if c in df.columns), None)
    if subject_col is None or task_col is None:
        return _empty(path, "Subject and task columns are both required for this coverage matrix.", "Subject × task coverage")
    work = df[[subject_col, task_col]].dropna().copy()
    if work.empty:
        return _empty(path, "No non-missing subject/task pairs were available.", "Subject × task coverage")
    top_subjects = work[subject_col].astype(str).value_counts().head(60).index.tolist()
    top_tasks = work[task_col].astype(str).value_counts().head(25).index.tolist()
    mat = pd.crosstab(work[subject_col].astype(str), work[task_col].astype(str)).reindex(index=top_subjects, columns=top_tasks, fill_value=0)
    fig, ax = plt.subplots(figsize=(max(8, 0.34 * len(top_tasks) + 4), max(6, 0.12 * len(top_subjects) + 3)))
    im = ax.imshow(mat.to_numpy(), aspect="auto", interpolation="nearest", cmap="viridis")
    ax.set_title("Subject × task recording coverage", fontsize=14, fontweight="bold", color=NAVY, pad=12)
    ax.set_xlabel("Task", color=MUTED); ax.set_ylabel("Subject", color=MUTED)
    ax.set_xticks(range(len(top_tasks))); ax.set_xticklabels(top_tasks, rotation=70, ha="right", fontsize=7, color=MUTED)
    ystep = max(1, len(top_subjects) // 30)
    yticks = list(range(0, len(top_subjects), ystep))
    ax.set_yticks(yticks); ax.set_yticklabels([top_subjects[i] for i in yticks], fontsize=6, color=MUTED)
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("records", color=MUTED)
    cbar.ax.tick_params(labelsize=8, colors=MUTED)
    return _save(fig, path)
