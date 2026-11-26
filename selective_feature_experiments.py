import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde, spearmanr
from transformers import HfArgumentParser

from summeval_data import METRICS, MODEL_CONFIGS
from feature_activation_analysis import JUDGES
from utils import load_pt

EPSILON = 1e-8

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Helvetica"],
        "font.size": 14,
        "axes.titlesize": 20,
        "axes.labelsize": 18,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "grid.linewidth": 0.4,
        "lines.linewidth": 1.5,
        "lines.markersize": 5,
        # Embed TrueType (Type 42) fonts — required by IEEE/ACM/arXiv pipelines.
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)
sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)

# Okabe-Ito palette (colorblind-safe, print-safe)
COLOR_ACTIVE = "#0072B2"
COLOR_SELECTIVE = "#CC79A7"
COLOR_INACTIVE = "#999999"
COLOR_NONSEL = "#E69F00"
COLOR_HIGH = "#009E73"
COLOR_LOW = "#D55E00"
LABEL_FONT_SIZE = 20

LEGEND_KWARGS = dict(
    frameon=True,
    framealpha=0.9,
    edgecolor="#cccccc",
    fancybox=True,
    fontsize=8,
    markerscale=2,
    loc="upper right",
)

HEATMAP_ANNOT_KWS = {"size": 9, "fontweight": "bold"}


def _fmt_rho(r: float) -> str:
    return f"ρ = {r:.2f}".replace("-", "−")


def _plot_jaccard_line(ax, x, vals, color, label):
    ax.plot(
        x,
        vals,
        color=color,
        linewidth=2.2,
        marker="o",
        markersize=8,
        markeredgecolor="white",
        markeredgewidth=1.2,
        label=label,
        zorder=3,
    )


def _style_matryoshka_ax(ax, bucket_x, ylabel):
    ax.set_xlabel(
        r"Feature index bin (low $\rightarrow$ high)",
        fontsize=23,
        fontweight="bold",
    )
    ax.set_ylabel(ylabel, fontsize=23, fontweight="bold")
    ax.set_xticks(bucket_x)
    ax.set_xticklabels([str(i) for i in bucket_x], fontsize=21)
    ax.tick_params(axis="y", labelsize=21)
    ax.set_ylim(0, 1.0)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncols=2,
        fontsize=18,
        columnspacing=1.2,
        frameon=True,
        framealpha=0.95,
        edgecolor="#cccccc",
        fancybox=False,
        borderpad=0.7,
        handlelength=1.6,
    )
    ax.grid(False)
    ax.grid(axis="y", linestyle=(0, (1, 2)), linewidth=0.9, color="#c8c8c8")
    ax.set_axisbelow(True)


def _get_dirs(model: str, analysis_dir=None, output_dir=None):
    output_base = (
        Path(output_dir).expanduser()
        if output_dir is not None
        else Path(f"activations_results/{model}/selective_feature_exp")
    )
    analysis_base = (
        Path(analysis_dir).expanduser()
        if analysis_dir is not None
        else Path(f"activations_results/{model}/feature_activation_analysis")
    )
    feature_ids_dir = analysis_base / "feature_ids"
    return output_base, feature_ids_dir


def _judge_plot_path(output_dir, filename, judge, args):
    path = output_dir / filename
    if args.judge == "both":
        path = path.with_name(f"{path.stem}_{judge}{path.suffix}")
    return path


@dataclass
class SelectiveExpArgs:
    model: str = field(
        default="gemma3",
        metadata={"help": "Model to analyze", "choices": ["gemma3", "qwen35"]},
    )
    judge: str = field(
        default="both",
        metadata={"help": "Judge to use", "choices": ["gemma", "expert", "both"]},
    )
    layers: list[int] | None = field(default=None)
    widths: list[str] | None = field(default=None)
    focus_layer: int = 53
    focus_width: str = "16k"
    cohen_d_thres: float = 0.5
    analysis_dir: str | None = field(
        default=None,
        metadata={"help": "Read split_summary.csv and feature_ids from this directory."},
    )
    sae_dir: str = field(
        default="results",
        metadata={"help": "Read SAE feature tensors from this directory."},
    )
    cohen_d_dir: str | None = field(
        default=None,
        metadata={"help": "Read Cohen's d JSON files from this directory."},
    )
    output_dir: str | None = field(
        default=None,
        metadata={"help": "Save experiment plots in this directory."},
    )
    experiment: int | None = field(
        default=None,
        metadata={
            "help": "Select one experiment. The default runs all experiments with available Cohen's d inputs.",
            "choices": list(range(1, 13)),
        },
    )


def _get_judges(args):
    if args.judge == "both":
        return list(JUDGES)
    return [args.judge]


def load_split_summary(analysis_dir):
    path = Path(analysis_dir) / "split_summary.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"split_summary.csv not found at {path}\n"
            "Run feature_activation_analysis.py first."
        )
    df = pd.read_csv(path)
    df["frac_selective"] = df["frac_selective"].astype(float)
    return df


def load_selective_ids(metric, judge, layer, width, feature_ids_dir):
    path = feature_ids_dir / f"{metric}_{judge}_L{layer}_W{width}_selective_ids.npy"
    if path.exists():
        return set(np.load(path).tolist())
    return set()


def load_active_ids(metric, judge, layer, width, feature_ids_dir):
    path = feature_ids_dir / f"{metric}_{judge}_L{layer}_W{width}_active_ids.npy"
    if path.exists():
        return set(np.load(path).tolist())
    return set()


def _cohen_d_path(layer, width, model, cohen_d_dir=None):
    directory = (
        Path(cohen_d_dir).expanduser()
        if cohen_d_dir is not None
        else Path("results") / "activated_idxes" / model
    )
    return directory / f"layer{layer}_width{width}.json"


def load_cohen_d(layer, width, model, cohen_d_dir=None):
    path = _cohen_d_path(layer, width, model, cohen_d_dir)
    if not path.exists():
        raise FileNotFoundError(
            f"Cohen's d file is missing: {path}. Set --cohen_d_dir to its directory."
        )
    with open(path) as f:
        return json.load(f)


def load_stats(metric, judge, layer, width, feature_ids_dir):
    """Load precomputed stats from feature_activation_analysis .npy files."""
    prefix = f"{metric}_{judge}_L{layer}_W{width}"
    d = feature_ids_dir
    selectivity_path = d / f"{prefix}_selectivity.npy"
    if not selectivity_path.exists():
        return None
    selectivity = np.load(selectivity_path)
    n_feat = len(selectivity)
    active_mask = np.load(d / f"{prefix}_active_mask.npy").astype(bool)
    selective_mask = np.load(d / f"{prefix}_selective_mask.npy").astype(bool)
    selective_high_mask = np.load(d / f"{prefix}_selective_high_ids.npy")
    selective_low_mask = np.load(d / f"{prefix}_selective_low_ids.npy")
    u_high = np.load(d / f"{prefix}_u_high.npy")
    u_low = np.load(d / f"{prefix}_u_low.npy")
    diff = np.abs(u_high - u_low)
    return {
        "selectivity": selectivity,
        "active_mask": active_mask,
        "selective_mask": selective_mask,
        "selective_high_ids": selective_high_mask,
        "selective_low_ids": selective_low_mask,
        "u_high": u_high,
        "u_low": u_low,
        "diff": diff,
        "n_feat": n_feat,
    }


def plot_selective_counts(args):
    output_base, feature_ids_dir = _get_dirs(args.model, args.analysis_dir, args.output_dir)
    config = MODEL_CONFIGS[args.model]
    out = output_base / "counts"
    out.mkdir(parents=True, exist_ok=True)

    summary = load_split_summary(feature_ids_dir.parent)
    layers = args.layers or config["layers"]
    widths = args.widths or config["widths"]
    judges = _get_judges(args)

    ymax_n = summary[["n_active", "n_selective"]].max().max() * 1.05
    ymax_frac = min(summary[["frac_active", "frac_selective"]].max().max() * 1.05, 1.0)

    width_colors = dict(zip(widths, [COLOR_ACTIVE, COLOR_NONSEL, COLOR_SELECTIVE]))
    if len(widths) > 3:
        extra = [COLOR_HIGH, COLOR_LOW]
        for i, w in enumerate(widths[3:], start=3):
            width_colors[w] = extra[i % len(extra)]

    n_judges = len(judges)
    n_metrics = len(METRICS)

    # Font sizes chosen for a full-\textwidth appendix figure (the 16in canvas
    # scales down to ~44%, landing these at roughly paper text size).
    _title_fs = 22
    _label_fs = 20
    _tick_fs = 17
    _legend_fs = 17

    for act_col, sel_col, ylab, ymax, fname in [
        (
            "n_active",
            "n_selective",
            "Number of features",
            ymax_n,
            "n_active_selective_line.pdf",
        ),
        (
            "frac_active",
            "frac_selective",
            "Fraction of features",
            ymax_frac,
            "frac_active_selective_line.pdf",
        ),
    ]:
        fig, axes = plt.subplots(
            n_judges,
            n_metrics,
            figsize=(4 * n_metrics, 3.6 * n_judges),
            sharey=True,
            sharex=True,
            squeeze=False,
        )
        for j, judge in enumerate(judges):
            for col, metric in enumerate(METRICS):
                ax = axes[j][col]
                for width in widths:
                    c = width_colors[width]
                    sub = summary[
                        (summary["metric"] == metric)
                        & (summary["judge"] == judge)
                        & (summary["width"] == width)
                    ].sort_values("layer")
                    ax.plot(
                        sub["layer"],
                        sub[act_col],
                        color=c,
                        linestyle=":",
                        linewidth=2.2,
                        marker="o",
                        markersize=7.5,
                        markerfacecolor="white",
                        markeredgecolor=c,
                        markeredgewidth=1.5,
                        label=f"{width} · Active",
                        zorder=3,
                    )
                    ax.plot(
                        sub["layer"],
                        sub[sel_col],
                        color=c,
                        linestyle="-",
                        linewidth=2.2,
                        marker="o",
                        markersize=7.5,
                        markerfacecolor=c,
                        markeredgecolor="white",
                        markeredgewidth=1.2,
                        label=f"{width} · Selective",
                        zorder=3,
                    )
                if j == 0:
                    ax.set_title(
                        metric.capitalize(),
                        fontsize=_title_fs,
                        fontweight="bold",
                        pad=8,
                    )
                if j == n_judges - 1:
                    ax.set_xlabel("Layer", fontsize=_label_fs, fontweight="bold")
                ax.set_xticks(layers)
                ax.tick_params(labelsize=_tick_fs)
                ax.set_ylim(0, ymax)
                ax.grid(False)
                ax.grid(axis="y", linestyle=(0, (1, 2)), linewidth=0.9, color="#c8c8c8")
                ax.set_axisbelow(True)
                if col == 0:
                    row_lab = ylab if n_judges == 1 else f"{ylab}\n({judge} judge)"
                    ax.set_ylabel(row_lab, fontsize=_label_fs, fontweight="bold")
                    if ylab.startswith("Fraction"):
                        # Fractions are well under 1%; pick decimals to match.
                        dec = 0 if ymax >= 0.05 else (1 if ymax >= 0.005 else 2)
                        ax.yaxis.set_major_formatter(
                            plt.FuncFormatter(lambda y, _, d=dec: f"{y:.{d}%}")
                        )

        handles, lbls = axes[0][0].get_legend_handles_labels()
        fig.legend(
            handles,
            lbls,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.0),
            ncols=max(1, len(handles) // 2),
            fontsize=_legend_fs,
            frameon=False,
            columnspacing=1.6,
            handlelength=1.8,
        )
        legend_frac = 0.82 if n_judges == 1 else 0.91
        fig.tight_layout(rect=(0, 0, 1, legend_frac))
        fig.savefig(out / fname)
        plt.close(fig)

    for judge in judges:
        for metric in METRICS:
            sub = summary[(summary["metric"] == metric) & (summary["judge"] == judge)]
            pivot = sub.pivot_table(
                index="width", columns="layer", values="n_selective", aggfunc="first"
            )
            pivot = pivot.reindex(widths)
            pivot = pivot[[layer for layer in layers if layer in pivot.columns]]

            fig, ax = plt.subplots(figsize=(5.4, 1.1 + 0.9 * len(widths)))
            sns.heatmap(
                pivot.astype(float),
                annot=True,
                fmt=".0f",
                cmap="Blues",
                vmin=0,
                ax=ax,
                linewidths=2,
                linecolor="white",
                annot_kws={"size": 14, "fontweight": "bold"},
                cbar_kws={"shrink": 0.9, "pad": 0.02},
            )
            ax.set_title(
                f"{metric.capitalize()} — {judge} judge",
                fontsize=15,
                fontweight="bold",
                pad=8,
            )
            ax.set_xlabel("Layer", fontsize=14, fontweight="bold")
            ax.set_ylabel("SAE width", fontsize=14, fontweight="bold")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=13)
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=13)
            ax.tick_params(length=0)
            cbar = ax.collections[0].colorbar
            cbar.ax.tick_params(labelsize=12)
            cbar.set_label("Selective features", fontsize=13, fontweight="bold")
            cbar.outline.set_edgecolor("#bbbbbb")
            cbar.outline.set_linewidth(0.8)
            fig.tight_layout()
            fname = (
                f"{metric}_heatmap.pdf"
                if n_judges == 1
                else f"{metric}_{judge}_heatmap.pdf"
            )
            fig.savefig(out / fname)
            plt.close(fig)

    print(f"[Experiment 1] Plots saved to {out}/")


def plot_inter_metric_overlap(args):
    output_base, feature_ids_dir = _get_dirs(args.model, args.analysis_dir, args.output_dir)
    out = output_base / "overlap"
    out.mkdir(parents=True, exist_ok=True)

    layer = args.focus_layer
    width = args.focus_width

    for judge in _get_judges(args):
        selective_sets = {}
        for metric in METRICS:
            selective_sets[metric] = load_selective_ids(
                metric, judge, layer, width, feature_ids_dir
            )

        n = len(METRICS)
        jaccard_mat = np.ones((n, n))
        intersection_mat = np.zeros((n, n))

        for i, m1 in enumerate(METRICS):
            for j, m2 in enumerate(METRICS):
                if i == j:
                    jaccard_mat[i, j] = 1.0
                    intersection_mat[i, j] = len(selective_sets[m1])
                else:
                    inter = len(selective_sets[m1] & selective_sets[m2])
                    union = len(selective_sets[m1] | selective_sets[m2])
                    jaccard_mat[i, j] = inter / union if union > 0 else 0.0
                    intersection_mat[i, j] = inter

        labels = [m.capitalize() for m in METRICS]

        fig, ax = plt.subplots(figsize=(5, 3))
        sns.heatmap(
            pd.DataFrame(jaccard_mat, index=labels, columns=labels),
            annot=True,
            fmt=".3f",
            cmap="cividis",
            vmin=0,
            vmax=1,
            ax=ax,
            linewidths=0.5,
            linecolor="white",
            annot_kws=HEATMAP_ANNOT_KWS,
            cbar_kws={"shrink": 0.8, "label": "Jaccard"},
        )
        plt.setp(ax.get_yticklabels(), rotation=30, ha="right")
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
        fig.tight_layout()
        fig.savefig(_judge_plot_path(out, f"jaccard.pdf", judge, args))
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(5, 3))
        sns.heatmap(
            pd.DataFrame(intersection_mat, index=labels, columns=labels),
            annot=True,
            fmt=".0f",
            cmap="YlOrRd",
            ax=ax,
            linewidths=0.5,
            linecolor="white",
            annot_kws=HEATMAP_ANNOT_KWS,
            cbar_kws={"shrink": 0.8, "label": "Count"},
        )
        plt.setp(ax.get_yticklabels(), rotation=30, ha="right")
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
        fig.tight_layout()
        fig.savefig(_judge_plot_path(out, f"intersection.pdf", judge, args))
        plt.close(fig)

    print(f"[Experiment 2] Plots saved to {out}/")


def plot_index_distribution(args):
    output_base, feature_ids_dir = _get_dirs(args.model, args.analysis_dir, args.output_dir)
    out = output_base / "index_dist"
    out.mkdir(parents=True, exist_ok=True)

    layer = args.focus_layer
    width = args.focus_width

    for metric in METRICS:
        for judge in _get_judges(args):
            st = load_stats(metric, judge, layer, width, feature_ids_dir)
            if st is None:
                print(f"  Skipping {metric}/{judge} (no saved results)")
                continue

            selectivity = st["selectivity"]
            selective_mask = st["selective_mask"]
            active_mask = st["active_mask"]
            indices = np.arange(st["n_feat"])

            corr_sel = spearmanr(
                indices[active_mask], selectivity[active_mask]
            ).statistic
            is_active = active_mask.astype(float)
            corr_act = spearmanr(indices, is_active).statistic

            fig, ax = plt.subplots(figsize=(8, 4))
            ns_x = indices[active_mask & ~selective_mask]
            ns_y = selectivity[active_mask & ~selective_mask]
            if len(ns_x) > 100:
                try:
                    ns_xy = np.vstack([ns_x, ns_y])
                    ns_kde = gaussian_kde(ns_xy)(ns_xy)
                    idx_sort = ns_kde.argsort()
                    ax.scatter(
                        ns_x[idx_sort],
                        ns_y[idx_sort],
                        c=ns_kde[idx_sort],
                        s=2,
                        alpha=0.15,
                        cmap="Blues",
                        edgecolors="none",
                        rasterized=True,
                    )
                except np.linalg.LinAlgError:
                    pass
            ax.scatter(
                indices[active_mask & ~selective_mask],
                selectivity[active_mask & ~selective_mask],
                s=6,
                alpha=0.15,
                color=COLOR_ACTIVE,
                label="Non-selective",
                edgecolors="none",
            )
            ax.scatter(
                indices[selective_mask],
                selectivity[selective_mask],
                s=14,
                alpha=0.8,
                color=COLOR_SELECTIVE,
                label="Selective",
                edgecolors="none",
            )
            ax.set_title(
                f"rho(index, selectivity) = {corr_sel:.3f},  "
                f"rho(index, active) = {corr_act:.3f}"
            )
            ax.set_xlabel("Feature index")
            ax.set_ylabel("Selectivity")
            ax.legend(**LEGEND_KWARGS)
            fig.tight_layout()
            fig.savefig(_judge_plot_path(out, f"{metric}_scatter.pdf", judge, args))
            plt.close(fig)

            sel_vals = selectivity[selective_mask]
            if len(sel_vals) > 0:
                fig, ax = plt.subplots(figsize=(5, 3))
                ax.hist(
                    sel_vals,
                    bins=40,
                    color=COLOR_SELECTIVE,
                    edgecolor="white",
                    linewidth=0.3,
                    alpha=0.85,
                )
                ax.set_title(f"n = {len(sel_vals)}")
                ax.set_xlabel("Selectivity")
                ax.set_ylabel("Count")
                fig.tight_layout()
                fig.savefig(_judge_plot_path(out, f"{metric}_hist.pdf", judge, args))
                plt.close(fig)

    print(f"[Experiment 3] Plots saved to {out}/")


def plot_selectivity_vs_cohens_d(args):
    output_base, feature_ids_dir = _get_dirs(args.model, args.analysis_dir, args.output_dir)
    out = output_base / "vs_cohens_d"
    out.mkdir(parents=True, exist_ok=True)

    layer = args.focus_layer
    width = args.focus_width

    cohen_d_data = load_cohen_d(layer, width, args.model, args.cohen_d_dir)

    for metric in METRICS:
        cohen_d = np.array(cohen_d_data[metric]["cohen_d"])
        cohen_d_abs = np.abs(cohen_d)

        for judge in _get_judges(args):
            st = load_stats(metric, judge, layer, width, feature_ids_dir)
            if st is None:
                print(f"  Skipping {metric}/{judge} (no saved results)")
                continue

            selectivity = st["selectivity"]
            selective_mask = st["selective_mask"]
            cohen_selective = cohen_d_abs > args.cohen_d_thres

            both = selective_mask & cohen_selective
            selective_only = selective_mask & ~cohen_selective
            cohen_only = ~selective_mask & cohen_selective

            corr = spearmanr(cohen_d_abs, selectivity).statistic

            fig, ax = plt.subplots(figsize=(6, 5))
            ax.scatter(
                cohen_d_abs[cohen_only],
                selectivity[cohen_only],
                s=8,
                alpha=0.6,
                color=COLOR_SELECTIVE,
                label=f"Cohen's d only ({cohen_only.sum()})",
            )
            ax.scatter(
                cohen_d_abs[selective_only],
                selectivity[selective_only],
                s=8,
                alpha=0.6,
                color=COLOR_SELECTIVE,
                label=f"Selective only ({selective_only.sum()})",
            )
            ax.scatter(
                cohen_d_abs[both],
                selectivity[both],
                s=14,
                alpha=0.8,
                color=COLOR_HIGH,
                label=f"Both ({both.sum()})",
            )
            ax.set_title(f"{metric} ({judge})  —  Spearman rho = {corr:.3f}")
            ax.set_xlabel("|Cohen's d|")
            ax.set_ylabel("Selectivity")
            ax.legend(**LEGEND_KWARGS)
            fig.tight_layout()
            fig.savefig(_judge_plot_path(out, f"{metric}_scatter.pdf", judge, args))
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(4, 3))
            cats = ["Both", "Cohen's d only", "Selective only"]
            counts = [both.sum(), cohen_only.sum(), selective_only.sum()]
            colors = [COLOR_HIGH, COLOR_NONSEL, COLOR_SELECTIVE]
            bars = ax.bar(
                cats,
                counts,
                color=colors,
                edgecolor="white",
                linewidth=0.8,
                width=0.6,
            )
            for bar in bars:
                h = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    h + 0.5,
                    f"{int(h)}",
                    ha="center",
                    va="bottom",
                    fontsize=16,
                    fontweight="medium",
                )
            ax.set_title(f"{metric} ({judge}) — Feature Selection Overlap")
            ax.set_ylabel("Number of features")
            fig.tight_layout()
            fig.savefig(_judge_plot_path(out, f"{metric}_overlap_bar.pdf", judge, args))
            plt.close(fig)

    print(f"[Experiment 4] Plots saved to {out}/")


def plot_matryoshka_metrics(args):
    output_base, feature_ids_dir = _get_dirs(args.model, args.analysis_dir, args.output_dir)
    out = output_base / "matryoshka_metrics"
    out.mkdir(parents=True, exist_ok=True)

    layer = args.focus_layer
    width = args.focus_width
    n_bins = 5
    judges = _get_judges(args)

    for judge in judges:
        active_vecs = {}
        selective_vecs = {}
        for metric in METRICS:
            st = load_stats(metric, judge, layer, width, feature_ids_dir)
            if st is None:
                print(f"  Skipping {metric}/{judge} (no saved results)")
                continue
            active_vecs[metric] = st["active_mask"]
            selective_vecs[metric] = st["selective_mask"]

        valid_metrics = list(selective_vecs.keys())
        if len(valid_metrics) < 2:
            print(
                f"  {judge}: not enough metrics with valid selectivity data, skipping."
            )
            continue

        n_features = len(active_vecs[valid_metrics[0]])
        bin_edges = np.linspace(0, n_features, n_bins + 1, dtype=int)
        bin_centers = [(bin_edges[b] + bin_edges[b + 1]) / 2 for b in range(n_bins)]

        from itertools import combinations

        active_jaccard = {
            pair: [] for pair in combinations(range(len(valid_metrics)), 2)
        }
        selective_jaccard = {
            pair: [] for pair in combinations(range(len(valid_metrics)), 2)
        }

        for b in range(n_bins):
            lo, hi = bin_edges[b], bin_edges[b + 1]
            for pair in active_jaccard:
                i, j = pair
                a_i, a_j = (
                    active_vecs[valid_metrics[i]][lo:hi],
                    active_vecs[valid_metrics[j]][lo:hi],
                )
                s_i, s_j = (
                    selective_vecs[valid_metrics[i]][lo:hi],
                    selective_vecs[valid_metrics[j]][lo:hi],
                )

                a_union = a_i | a_j
                a_inter = a_i & a_j
                active_jaccard[pair].append(a_inter.sum() / (a_union.sum() + EPSILON))

                s_union = s_i | s_j
                s_inter = s_i & s_j
                selective_jaccard[pair].append(
                    s_inter.sum() / (s_union.sum() + EPSILON)
                )

        bucket_x = list(range(1, n_bins + 1))

        for pair in active_jaccard:
            m1, m2 = valid_metrics[pair[0]], valid_metrics[pair[1]]
            rho_active = spearmanr(bin_centers, active_jaccard[pair]).statistic
            rho_selective = spearmanr(bin_centers, selective_jaccard[pair]).statistic

            fig, ax = plt.subplots(figsize=(7.2, 4.6))
            _plot_jaccard_line(
                ax,
                bucket_x,
                active_jaccard[pair],
                COLOR_ACTIVE,
                f"Active ({_fmt_rho(rho_active)})",
            )
            _plot_jaccard_line(
                ax,
                bucket_x,
                selective_jaccard[pair],
                COLOR_SELECTIVE,
                f"Selective ({_fmt_rho(rho_selective)})",
            )
            _style_matryoshka_ax(ax, bucket_x, "Jaccard overlap")
            fig.tight_layout()
            fig.savefig(_judge_plot_path(out, f"{m1}_{m2}_jaccard.pdf", judge, args))
            plt.close(fig)

        active_avg = np.mean([active_jaccard[p] for p in active_jaccard], axis=0)
        selective_avg = np.mean(
            [selective_jaccard[p] for p in selective_jaccard], axis=0
        )
        rho_active_avg = spearmanr(bin_centers, active_avg).statistic
        rho_selective_avg = spearmanr(bin_centers, selective_avg).statistic

        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        _plot_jaccard_line(
            ax,
            bucket_x,
            active_avg,
            COLOR_ACTIVE,
            f"Active ({_fmt_rho(rho_active_avg)})",
        )
        _plot_jaccard_line(
            ax,
            bucket_x,
            selective_avg,
            COLOR_SELECTIVE,
            f"Selective ({_fmt_rho(rho_selective_avg)})",
        )
        _style_matryoshka_ax(ax, bucket_x, "Mean Jaccard overlap")
        fig.tight_layout()
        fig.savefig(_judge_plot_path(out, f"mean_jaccard.pdf", judge, args))
        plt.close(fig)

    print(f"[Experiment 5] Plots saved to {out}/")


def plot_selective_vs_active_profiles(args):
    output_base, feature_ids_dir = _get_dirs(args.model, args.analysis_dir, args.output_dir)
    out = output_base / "selective_profiles"
    out.mkdir(parents=True, exist_ok=True)

    layer = args.focus_layer
    width = args.focus_width

    for metric in METRICS:
        for judge in _get_judges(args):
            st = load_stats(metric, judge, layer, width, feature_ids_dir)
            if st is None:
                print(f"  Skipping {metric}/{judge} (no saved results)")
                continue

            sel_mask = st["selective_mask"]
            act_mask = st["active_mask"]
            nonselective_active = act_mask & ~sel_mask
            diff = st["diff"]

            selective_diff = diff[sel_mask]
            nonsel_diff = diff[nonselective_active]

            if len(selective_diff) == 0 or len(nonsel_diff) == 0:
                print(f"  {metric}/{judge}: empty group, skipping")
                continue

            fig, ax = plt.subplots(figsize=(6, 4))
            bins = np.histogram_bin_edges(
                np.concatenate([selective_diff, nonsel_diff]), bins=40
            )
            ax.hist(
                selective_diff,
                bins=bins,
                alpha=0.65,
                color=COLOR_SELECTIVE,
                label=f"Selective (n={len(selective_diff)})",
                density=True,
                edgecolor="white",
                linewidth=0.3,
            )
            ax.hist(
                nonsel_diff,
                bins=bins,
                alpha=0.65,
                color=COLOR_ACTIVE,
                label=f"Active Non-selective (n={len(nonsel_diff)})",
                density=True,
                edgecolor="white",
                linewidth=0.3,
            )
            ax.axvline(
                np.median(selective_diff),
                color=COLOR_SELECTIVE,
                linestyle="--",
                linewidth=1.5,
                label=f"Sel. median: {np.median(selective_diff):.4f}",
            )
            ax.axvline(
                np.median(nonsel_diff),
                color=COLOR_ACTIVE,
                linestyle="--",
                linewidth=1.5,
                label=f"Non-sel. median: {np.median(nonsel_diff):.4f}",
            )
            ax.set_xlabel("u_high - u_low (mean activation difference)")
            ax.set_ylabel("Density")
            ax.set_title(f"{metric} ({judge}) — L{layer} W{width}")
            ax.legend(fontsize=8)
            ax.grid(axis="y", alpha=0.3)
            fig.tight_layout()
            fig.savefig(_judge_plot_path(out, f"{metric}_diff_hist.pdf", judge, args))
            plt.close(fig)

    print(f"[Experiment 6] Plots saved to {out}/")


def plot_selectivity_breadth(args):
    output_base, feature_ids_dir = _get_dirs(args.model, args.analysis_dir, args.output_dir)
    out = output_base / "selectivity_breadth"
    out.mkdir(parents=True, exist_ok=True)

    layer = args.focus_layer
    width = args.focus_width

    for judge in _get_judges(args):
        sel_sets = {}
        for metric in METRICS:
            sel_sets[metric] = load_selective_ids(
                metric, judge, layer, width, feature_ids_dir
            )

        all_ids = set()
        for s in sel_sets.values():
            all_ids.update(s)
        if not all_ids:
            print(f"  {judge}: no selective features found, skipping.")
            continue

        all_ids = sorted(all_ids)
        breadth = np.zeros(len(all_ids), dtype=int)
        for idx, fid in enumerate(all_ids):
            for metric in METRICS:
                if fid in sel_sets[metric]:
                    breadth[idx] += 1

        counts = np.bincount(breadth, minlength=len(METRICS) + 1)
        bins = np.arange(1, len(METRICS) + 1)
        counts = counts[1:]
        frac_breadth = counts / counts.sum()

        median_fids = []
        for b in bins:
            ids_in_bin = np.array(all_ids)[breadth == b]
            median_fids.append(int(np.median(ids_in_bin)) if len(ids_in_bin) > 0 else 0)

        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(
            bins,
            frac_breadth,
            color=COLOR_SELECTIVE,
            edgecolor="white",
            linewidth=0.8,
            width=0.6,
        )
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.005,
                f"{h:.1%}",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="medium",
            )
        ax.set_xlabel("Number of metrics selective for")
        ax.set_ylabel("Fraction of selective features")
        ax.set_title(f"n={counts.sum()}")
        ax.set_xticks(bins)
        ax.set_xticklabels([str(b) for b in bins])
        fig.tight_layout()
        fig.savefig(_judge_plot_path(out, f"breadth.pdf", judge, args))
        plt.close(fig)

    print(f"[Experiment 7] Plots saved to {out}/")


def plot_asymmetric_overlap(args):
    output_base, feature_ids_dir = _get_dirs(args.model, args.analysis_dir, args.output_dir)
    out = output_base / "asymmetric_overlap"
    out.mkdir(parents=True, exist_ok=True)

    layer = args.focus_layer
    width = args.focus_width

    for judge in _get_judges(args):
        sel_sets = {}
        for metric in METRICS:
            sel_sets[metric] = load_selective_ids(
                metric, judge, layer, width, feature_ids_dir
            )

        n = len(METRICS)
        labels = [m.capitalize() for m in METRICS]
        asymp_mat = np.zeros((n, n))
        counts_mat = np.zeros((n, n), dtype=int)

        for i, m1 in enumerate(METRICS):
            for j, m2 in enumerate(METRICS):
                s1 = sel_sets[m1]
                s2 = sel_sets[m2]
                if i == j:
                    asymp_mat[i, j] = 1.0
                    counts_mat[i, j] = len(s1)
                else:
                    if len(s1) > 0:
                        asymp_mat[i, j] = len(s1 & s2) / len(s1)
                    counts_mat[i, j] = len(s1 & s2)

        fig, ax = plt.subplots(figsize=(5, 3))
        sns.heatmap(
            pd.DataFrame(asymp_mat, index=labels, columns=labels),
            annot=True,
            fmt=".3f",
            cmap="cividis",
            vmin=0,
            vmax=1,
            ax=ax,
            linewidths=0.5,
            linecolor="white",
            annot_kws=HEATMAP_ANNOT_KWS,
            cbar_kws={"shrink": 0.8, "label": "P(m2 | m1)"},
        )
        ax.set_title(f"{judge} — Asymmetric Overlap\nP(m2 | m1)")
        ax.set_xlabel("Column metric (m2)")
        ax.set_ylabel("Row metric (m1)")
        plt.setp(ax.get_yticklabels(), rotation=30, ha="right")
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
        fig.tight_layout()
        fig.savefig(_judge_plot_path(out, f"asymmetric.pdf", judge, args))
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(5, 3))
        sns.heatmap(
            pd.DataFrame(counts_mat, index=labels, columns=labels),
            annot=True,
            fmt="d",
            cmap="YlOrRd",
            ax=ax,
            linewidths=0.5,
            linecolor="white",
            annot_kws=HEATMAP_ANNOT_KWS,
            cbar_kws={"shrink": 0.8, "label": "Count"},
        )
        ax.set_title(f"{judge} — Intersection Counts")
        plt.setp(ax.get_yticklabels(), rotation=30, ha="right")
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
        fig.tight_layout()
        fig.savefig(_judge_plot_path(out, f"asymmetric_counts.pdf", judge, args))
        plt.close(fig)

    print(f"[Experiment 8] Plots saved to {out}/")


def plot_selective_redundancy(args):
    output_base, feature_ids_dir = _get_dirs(args.model, args.analysis_dir, args.output_dir)
    config = MODEL_CONFIGS[args.model]
    out = output_base / "selective_redundancy"
    out.mkdir(parents=True, exist_ok=True)

    layer = args.focus_layer
    width = args.focus_width

    for metric in METRICS:
        for judge in _get_judges(args):
            st = load_stats(metric, judge, layer, width, feature_ids_dir)
            if st is None:
                print(f"  Skipping {metric}/{judge} (no saved results)")
                continue

            sel_ids = np.array(
                sorted(load_selective_ids(metric, judge, layer, width, feature_ids_dir))
            )
            act_ids = np.array(
                sorted(load_active_ids(metric, judge, layer, width, feature_ids_dir))
            )
            sel_mask = st["selective_mask"]
            act_mask = st["active_mask"]

            if len(sel_ids) < 3:
                print(
                    f"  {metric}/{judge}: {len(sel_ids)} selective features, skipping correlation."
                )
                continue

            sae_feat = (
                load_pt(
                    Path(args.sae_dir).expanduser()
                    / f"sae_features_{config['sae_prefix']}_{metric}_layer{layer}_width{width}_last_token.pt",
                    map_location="cpu",
                    weights_only=True,
                )
                .detach()
                .numpy()
            )

            feat_sel = sae_feat[:, sel_ids]
            corr_sel = np.corrcoef(feat_sel.T)
            triu_sel = np.triu_indices(len(sel_ids), k=1)
            corr_vals_sel = corr_sel[triu_sel]

            act_nonsel_ids = act_ids[~sel_mask[act_ids]]
            if len(act_nonsel_ids) >= 3:
                feat_act = sae_feat[:, act_nonsel_ids]
                corr_act = np.corrcoef(feat_act.T)
                triu_act = np.triu_indices(len(act_nonsel_ids), k=1)
                corr_vals_act = corr_act[triu_act]
            else:
                corr_vals_act = np.array([])

            fig, ax = plt.subplots(figsize=(6, 4))
            if len(corr_vals_act) > 0:
                ax.hist(
                    corr_vals_act,
                    bins=50,
                    color=COLOR_ACTIVE,
                    edgecolor="white",
                    linewidth=0.3,
                    alpha=0.65,
                    label=f"Active non-selective (n={len(act_nonsel_ids)})",
                    density=True,
                )
            ax.hist(
                corr_vals_sel,
                bins=50,
                color=COLOR_SELECTIVE,
                edgecolor="white",
                linewidth=0.3,
                alpha=0.7,
                label=f"Selective (n={len(sel_ids)})",
                density=True,
            )
            ax.axvline(0, color="black", linestyle="--", linewidth=0.5)
            ax.set_xlabel("Pairwise Pearson r")
            ax.set_ylabel("Density")

            def _iqr_str(vals):
                q1, q3 = np.percentile(vals, [25, 75])
                return f"med={np.median(vals):.3f}, IQR=[{q1:.3f}, {q3:.3f}]"

            bin_width_sel = (corr_vals_sel.max() - corr_vals_sel.min()) / 50
            stats_text = (
                f"Selective: {_iqr_str(corr_vals_sel)}\nBin width: {bin_width_sel:.4f}"
            )
            if len(corr_vals_act) > 0:
                bin_width_act = (corr_vals_act.max() - corr_vals_act.min()) / 50
                stats_text += f"\nActive: {_iqr_str(corr_vals_act)}\nBin width: {bin_width_act:.4f}"
            ax.text(
                0.95,
                0.95,
                stats_text,
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=7,
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            )
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(_judge_plot_path(out, f"{metric}_redundancy.pdf", judge, args))
            plt.close(fig)

    print(f"[Experiment 9] Plots saved to {out}/")


def plot_cohens_d_vs_selectivity_scatter(args):
    output_base, feature_ids_dir = _get_dirs(args.model, args.analysis_dir, args.output_dir)
    out = output_base / "cohens_d_scatter"
    out.mkdir(parents=True, exist_ok=True)

    layer = args.focus_layer
    width = args.focus_width

    cohen_d_data = load_cohen_d(layer, width, args.model, args.cohen_d_dir)

    for metric in METRICS:
        cohen_d = np.array(cohen_d_data[metric]["cohen_d"])
        cohen_d_abs = np.abs(cohen_d)

        for judge in _get_judges(args):
            st = load_stats(metric, judge, layer, width, feature_ids_dir)
            if st is None:
                print(f"  Skipping {metric}/{judge} (no saved results)")
                continue

            sel_mask = st["selective_mask"]
            act_mask = st["active_mask"]
            selectivity = st["selectivity"]

            inactive = ~act_mask
            active_nonsel = act_mask & ~sel_mask
            selective = sel_mask

            groups = [
                ("Selective", selective, COLOR_SELECTIVE, 14, 0.8),
                ("Active non-selective", active_nonsel, COLOR_NONSEL, 6, 0.3),
                ("Inactive", inactive, "lightgray", 3, 0.2),
            ]

            fig = plt.figure(figsize=(10, 8))
            gs = fig.add_gridspec(
                2,
                2,
                width_ratios=[3, 1],
                height_ratios=[1, 3],
                hspace=0.05,
                wspace=0.05,
            )

            ax_histx = fig.add_subplot(gs[0, 0])
            ax_main = fig.add_subplot(gs[1, 0])
            ax_histy = fig.add_subplot(gs[1, 1])

            for name, mask, color, sz, alpha in groups:
                x_vals = cohen_d_abs[mask]
                y_vals = selectivity[mask]
                if mask.sum() > 200:
                    try:
                        xy = np.vstack([x_vals, y_vals])
                        kde = gaussian_kde(xy)(xy)
                        idx_sort = kde.argsort()
                        ax_main.scatter(
                            x_vals[idx_sort],
                            y_vals[idx_sort],
                            c=kde[idx_sort],
                            s=1,
                            alpha=0.12,
                            cmap="Greys" if name == "Inactive" else "Blues",
                            edgecolors="none",
                            rasterized=True,
                        )
                    except np.linalg.LinAlgError:
                        pass

            for name, mask, color, sz, alpha in groups:
                ax_main.scatter(
                    cohen_d_abs[mask],
                    selectivity[mask],
                    s=sz,
                    alpha=alpha,
                    color=color,
                    label=f"{name} ({mask.sum()})",
                    edgecolors="none",
                )

            ax_main.set_xlabel("|Cohen's d|")
            ax_main.set_ylabel("Selectivity (lift)")
            ax_main.legend(fontsize=7, markerscale=2, loc="upper right")
            ax_main.grid(alpha=0.2)

            for name, mask, color, _, _ in groups:
                vals = cohen_d_abs[mask]
                vals = vals[vals > 0]
                if len(vals) > 0:
                    ax_histx.hist(
                        vals,
                        bins=50,
                        color=color,
                        alpha=0.7,
                        edgecolor="white",
                        linewidth=0.3,
                    )
            ax_histx.set_xlim(ax_main.get_xlim())
            ax_histx.tick_params(labelbottom=False)
            ax_histx.set_ylabel("Count", fontsize=8)

            for name, mask, color, _, _ in groups:
                vals = selectivity[mask]
                vals = vals[vals > 0]
                if len(vals) > 0:
                    ax_histy.hist(
                        vals,
                        bins=50,
                        color=color,
                        alpha=0.7,
                        edgecolor="white",
                        linewidth=0.3,
                        orientation="horizontal",
                    )
            ax_histy.set_ylim(ax_main.get_ylim())
            ax_histy.tick_params(labelleft=False)
            ax_histy.set_xlabel("Count", fontsize=8)

            corr = spearmanr(cohen_d_abs, selectivity).statistic
            fig.suptitle(
                f"{metric} ({judge}) — L{layer} W{width}  |  Spearman rho = {corr:.3f}",
                fontsize=11,
                y=0.98,
            )
            fig.savefig(_judge_plot_path(out, f"{metric}_scatter.pdf", judge, args))
            plt.close(fig)

    print(f"[Experiment 10] Plots saved to {out}/")


def plot_high_low_distributions(args):
    output_base, feature_ids_dir = _get_dirs(args.model, args.analysis_dir, args.output_dir)
    out = output_base / "high_low_distributions"
    out.mkdir(parents=True, exist_ok=True)

    layer = args.focus_layer
    width = args.focus_width

    for judge in _get_judges(args):
        n_highs = []
        n_lows = []
        med_highs = []
        med_lows = []
        labels = []
        for metric in METRICS:
            st = load_stats(metric, judge, layer, width, feature_ids_dir)
            if st is None:
                print(f"  Skipping {metric}/{judge} (no saved results)")
                continue
            n_highs.append(len(st["selective_high_ids"]))
            n_lows.append(len(st["selective_low_ids"]))
            med_highs.append(
                int(np.median(st["selective_high_ids"]))
                if len(st["selective_high_ids"]) > 0
                else 0
            )
            med_lows.append(
                int(np.median(st["selective_low_ids"]))
                if len(st["selective_low_ids"]) > 0
                else 0
            )
            labels.append(metric.capitalize())

        if not n_highs:
            print(f"  {judge}: no valid metrics, skipping")
            continue

        x = np.arange(len(labels))
        bar_w = 0.35

        fig, ax = plt.subplots(figsize=(7, 4))
        bars_high = ax.bar(
            x - bar_w / 2,
            n_highs,
            bar_w,
            color=COLOR_HIGH,
            label="Selective for high-scoring",
            edgecolor="white",
            linewidth=0.8,
        )
        bars_low = ax.bar(
            x + bar_w / 2,
            n_lows,
            bar_w,
            color=COLOR_LOW,
            label="Selective for low-scoring",
            edgecolor="white",
            linewidth=0.8,
        )
        for bar in list(bars_high) + list(bars_low):
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.3,
                f"{int(h)}",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="medium",
            )
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Number of selective features")
        ax.legend(**LEGEND_KWARGS)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(_judge_plot_path(out, f"high_low.pdf", judge, args))
        plt.close(fig)

    print(f"[Experiment 11] Plots saved to {out}/")


def plot_selective_activation_frequency(args):
    output_base, feature_ids_dir = _get_dirs(args.model, args.analysis_dir, args.output_dir)
    config = MODEL_CONFIGS[args.model]
    out = output_base / "selective_activation_freq"
    out.mkdir(parents=True, exist_ok=True)

    layer = args.focus_layer
    width = args.focus_width

    for metric in METRICS:
        for judge in _get_judges(args):
            st = load_stats(metric, judge, layer, width, feature_ids_dir)
            if st is None:
                print(f"  Skipping {metric}/{judge} (no saved results)")
                continue

            selective_ids = np.array(
                sorted(load_selective_ids(metric, judge, layer, width, feature_ids_dir))
            )
            if len(selective_ids) == 0:
                print(f"  {metric}/{judge}: no selective features, skipping")
                continue

            sae_feat = (
                load_pt(
                    Path(args.sae_dir).expanduser()
                    / f"sae_features_{config['sae_prefix']}_{metric}_layer{layer}_width{width}_last_token.pt",
                    map_location="cpu",
                    weights_only=True,
                )
                .detach()
                .numpy()
            )

            feat_sel = sae_feat[:, selective_ids]
            activation_counts = (feat_sel > 0).sum(axis=0)

            n_items = sae_feat.shape[0]

            fig, ax = plt.subplots(figsize=(7, 4))
            ax.hist(
                activation_counts,
                bins=min(50, len(selective_ids)),
                color=COLOR_SELECTIVE,
                edgecolor="white",
                alpha=0.85,
            )
            ax.axvline(
                np.median(activation_counts),
                color=COLOR_ACTIVE,
                linestyle="--",
                linewidth=1.5,
                label=f"Median: {np.median(activation_counts):.0f}",
            )
            ax.set_xlabel("Number of items activated on")
            ax.set_ylabel("Number of selective features")
            ax.set_title(
                f"{metric} ({judge}) — L{layer} W{width}\n"
                f"n={len(selective_ids)} selective features, {n_items} total items"
            )
            ax.legend(fontsize=8)
            ax.grid(axis="y", alpha=0.3)
            fig.tight_layout()
            fig.savefig(_judge_plot_path(out, f"{metric}_activation_freq.pdf", judge, args))
            plt.close(fig)

            print(
                f"  {metric}/{judge}: median={np.median(activation_counts):.0f}, "
                f"mean={np.mean(activation_counts):.1f}, "
                f"min={activation_counts.min()}, max={activation_counts.max()}"
            )

    print(f"[Experiment 12] Plots saved to {out}/")


def main():
    parser = HfArgumentParser(SelectiveExpArgs)
    args = parser.parse_args_into_dataclasses()[0]

    if args.focus_width == "16k" and args.model == "qwen35":
        args.focus_width = "80k"

    experiments = {
        1: plot_selective_counts,
        2: plot_inter_metric_overlap,
        3: plot_index_distribution,
        4: plot_selectivity_vs_cohens_d,
        5: plot_matryoshka_metrics,
        6: plot_selective_vs_active_profiles,
        7: plot_selectivity_breadth,
        8: plot_asymmetric_overlap,
        9: plot_selective_redundancy,
        10: plot_cohens_d_vs_selectivity_scatter,
        11: plot_high_low_distributions,
        12: plot_selective_activation_frequency,
    }
    for number, experiment in experiments.items():
        if args.experiment is not None and args.experiment != number:
            continue
        if args.experiment is None and number in (4, 10):
            path = _cohen_d_path(
                args.focus_layer, args.focus_width, args.model, args.cohen_d_dir
            )
            if not path.is_file():
                print(
                    f"Skipped experiment {number}: Cohen's d file is missing. "
                    "Set --cohen_d_dir to its directory."
                )
                continue
        experiment(args)


if __name__ == "__main__":
    main()
