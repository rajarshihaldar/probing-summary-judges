from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats as sp_stats
from scipy.special import betainc
from transformers import HfArgumentParser

from summeval_data import (
    METRICS,
    MODEL_CONFIGS,
    load_dataset,
)


def _get_output_dir(model: str) -> Path:
    return Path(f"activations_results/{model}/feature_activation_analysis")


EPSILON = 1e-8

JUDGES = ["gemma", "expert"]
THRESHOLDS = {
    "coherence": (3, 4),
    "consistency": (3, 4),
    "relevance": (3, 4),
    "fluency": (3, 4),
}

CI_ALPHA = 0.05
LIFT_THRESHOLD = 0.2  # Minimum |p_high - base_rate| for lift-based selectivity modes
M_MIN = 0
SELECTIVITY_MODES = [
    "lift_abs",
    "lift_signed",
    "lift_ratio",
    "cohen_h",
    "bayesian",
]
# Test backing the reliability filter (`use_ci`).  "wilson" keeps the normal
# approximation (Wilson score interval, plus a normal-approximation p-value in
# cohen_h mode); "fisher" replaces both with Fisher's exact test on the 2x2
# activation-by-split table.
SIGNIFICANCE_TESTS = ["wilson", "fisher"]
# Modes that carry their own significance criterion and so ignore `use_ci`.
_SELF_SIGNIFICANT_MODES = ["bayesian"]


@dataclass
class FeatureAnalysisArgs:
    model: str = field(
        default="gemma3",
        metadata={"help": "Model to analyze", "choices": ["gemma3", "qwen35"]},
    )
    layers: list[int] | None = field(default=None)
    widths: list[str] | None = field(default=None)
    metrics: list[str] | None = field(
        default=None,
        metadata={"help": "Select the metrics to analyze.", "choices": METRICS},
    )
    input_dir: str = field(
        default="results",
        metadata={"help": "Read SAE features from this directory."},
    )
    metadata_dir: str | None = field(
        default=None,
        metadata={
            "help": "Read metadata from this directory. The default is input_dir."
        },
    )
    output_dir: str | None = field(
        default=None,
        metadata={"help": "Save analysis files in this directory."},
    )
    m_min: float = field(
        default=M_MIN,
        metadata={
            "help": "Minimum mean activation (m_f) to consider a feature active (default: 0.0)"
        },
    )
    use_selectivity_threshold: bool = field(
        default=True,
        metadata={
            "help": "Apply selectivity threshold (mode-dependent, default: True)"
        },
    )
    selectivity_threshold: float = field(
        default=LIFT_THRESHOLD,
        metadata={
            "help": "Minimum threshold for selectivity (mode-dependent, default: 0.2)"
        },
    )
    use_ci: bool = field(
        default=False,
        metadata={
            "help": "Apply reliability filter (not applicable to bayesian, default: False)"
        },
    )
    significance_test: str = field(
        default="fisher",
        metadata={
            "help": f"Test backing the reliability filter: {SIGNIFICANCE_TESTS} (default: wilson)",
            "choices": SIGNIFICANCE_TESTS,
        },
    )
    ci_alpha: float = field(
        default=CI_ALPHA,
        metadata={
            "help": "Significance level for the reliability filter (default: 0.05)"
        },
    )
    selectivity_mode: str = field(
        default="cohen_h",
        metadata={"help": f"Selectivity mode: {SELECTIVITY_MODES} (default: cohen_h)"},
    )
    bayesian_prior_kappa: float = field(
        default=10.0,
        metadata={
            "help": "Prior concentration for Bayesian selectivity mode (default: 10.0)"
        },
    )


def get_split_indices(metric, judge, dataset):
    low_t, high_t = THRESHOLDS[metric]
    metadata = dataset.metadata[metric]
    high_idx, low_idx = [], []
    for i, md in enumerate(metadata):
        if judge == "gemma":
            score = md["decoded_score"]
        else:
            scores = [a[metric] for a in md["expert_annotations"]]
            score = round(float(np.median(scores)))
        if score >= high_t:
            high_idx.append(i)
        elif score <= low_t:
            low_idx.append(i)
    return np.array(high_idx), np.array(low_idx)


def wilson_ci_limits(count, n, alpha=0.05):
    """Wilson score confidence interval for a binomial proportion.

    Returns (lower, upper) bounds. Handles n=0 gracefully.
    """
    if n == 0:
        return 0.0, 1.0
    z = sp_stats.norm.ppf(1 - alpha / 2)
    p = count / n
    denominator = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denominator
    margin = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denominator
    return center - margin, center + margin


def _wilson_ci_condition(high_on, total_on, base_rate, ci_alpha, n_feat):
    """Vectorized Wilson CI check: True when CI excludes base_rate."""
    high_on_np = high_on.cpu().numpy().astype(np.float64)
    total_on_np = total_on.cpu().numpy().astype(np.float64)

    ci_low = np.zeros(n_feat, dtype=np.float64)
    ci_high = np.ones(n_feat, dtype=np.float64)

    nonzero = total_on_np > 0
    n = total_on_np[nonzero]
    c = high_on_np[nonzero]

    z = sp_stats.norm.ppf(1 - ci_alpha / 2)
    p = c / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom

    ci_low[nonzero] = center - margin
    ci_high[nonzero] = center + margin

    return torch.from_numpy((ci_low > base_rate) | (ci_high < base_rate)).bool()


def _to_numpy(x):
    return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)


def fisher_exact_pvalues(high_on, low_on, n_high, n_low, alternative="two-sided"):
    """Fisher's exact test over a batch of 2x2 tables, one per feature.

    Each feature contributes the contingency table

                     on        off
        high     high_on     n_high - high_on
        low       low_on     n_low  - low_on

    Under the null that firing is independent of the split, and conditioning on
    both margins, high_on ~ Hypergeom(M=n_high+n_low, n=n_high, N=high_on+low_on).
    Unlike the Wilson/z-score filters this makes no normal approximation, so it
    stays calibrated for the rare features where total_on is small -- which is
    most of them at wide SAE widths.

    ``alternative`` is "two-sided" (minimum-likelihood criterion, matching
    ``scipy.stats.fisher_exact``), "greater" (feature enriched in the high
    split), or "less".  Returns a float64 array of p-values.
    """
    high_on = _to_numpy(high_on).astype(np.int64).ravel()
    low_on = _to_numpy(low_on).astype(np.int64).ravel()
    total_on = high_on + low_on
    total = n_high + n_low

    if alternative == "greater":
        return sp_stats.hypergeom.sf(high_on - 1, total, n_high, total_on)
    if alternative == "less":
        return sp_stats.hypergeom.cdf(high_on, total, n_high, total_on)
    if alternative != "two-sided":
        raise ValueError(f"Unknown alternative: {alternative}")

    # Two-sided: sum the mass of every table no more likely than the observed
    # one.  Features sharing a column margin (total_on) share a null
    # distribution, so build each pmf once per distinct total_on rather than
    # once per feature -- there are at most n_high + n_low + 1 of them, versus
    # up to ~10^5 features.
    p_value = np.ones(high_on.shape[0], dtype=np.float64)
    for n_on in np.unique(total_on):
        idx = np.flatnonzero(total_on == n_on)
        lo = max(0, int(n_on) - n_low)
        hi = min(n_high, int(n_on))
        pmf = sp_stats.hypergeom.pmf(np.arange(lo, hi + 1), total, n_high, int(n_on))

        sorted_pmf = np.sort(pmf)
        cumulative = np.cumsum(sorted_pmf)
        # Relative slack so ties with the observed table are counted in.
        cutoff = pmf[high_on[idx] - lo] * (1 + 1e-7)
        pos = np.searchsorted(sorted_pmf, cutoff, side="right")
        p_value[idx] = np.where(pos > 0, cumulative[np.maximum(pos - 1, 0)], 0.0)

    return np.clip(p_value, 0.0, 1.0)


def compute_split_stats(
    sae_features,
    high_idx,
    low_idx,
    m_min,
    selectivity_threshold,
    use_selectivity_threshold,
    use_ci,
    ci_alpha,
    epsilon,
    selectivity_mode="lift_abs",
    bayesian_prior_kappa=10.0,
    significance_test="wilson",
):
    n_feat = sae_features.shape[1]

    if significance_test not in SIGNIFICANCE_TESTS:
        raise ValueError(
            f"Invalid significance_test: {significance_test}. Must be one of {SIGNIFICANCE_TESTS}"
        )

    if len(high_idx) == 0 or len(low_idx) == 0:
        return None

    n_high = len(high_idx)
    n_low = len(low_idx)
    total = n_high + n_low
    base_rate = n_high / total

    high_acts = sae_features[high_idx]
    low_acts = sae_features[low_idx]

    u_high = high_acts.mean(dim=0)
    u_low = low_acts.mean(dim=0)

    m_f = (u_high * n_high + u_low * n_low) / total
    active_mask = m_f > m_min
    n_active = active_mask.sum().item()

    high_on = (high_acts > 0).sum(dim=0)
    low_on = (low_acts > 0).sum(dim=0)
    total_on = high_on + low_on

    p_high = high_on / (total_on + epsilon)

    # cohen_h and the shared reliability filter both want Fisher p-values; the
    # tables don't change between them, so compute each alternative once.
    fisher_cache = {}

    def fisher_p(alternative):
        if alternative not in fisher_cache:
            fisher_cache[alternative] = fisher_exact_pvalues(
                high_on, low_on, n_high, n_low, alternative
            )
        return fisher_cache[alternative]

    is_directional = False

    if selectivity_mode == "lift_abs":
        selectivity = torch.abs(p_high - base_rate)
        is_significant = selectivity > selectivity_threshold

    elif selectivity_mode == "lift_signed":
        selectivity = p_high - base_rate
        is_significant = selectivity > selectivity_threshold
        is_directional = True

    elif selectivity_mode == "lift_ratio":
        selectivity = p_high / (base_rate + epsilon)
        is_significant = selectivity > (1.0 + selectivity_threshold)
        is_directional = True

    elif selectivity_mode == "cohen_h":
        h = 2 * (
            torch.asin(torch.sqrt(p_high))
            - torch.asin(torch.sqrt(torch.tensor(base_rate)))
        )
        selectivity = h
        if use_ci:
            if significance_test == "fisher":
                p_value_np = fisher_p("two-sided")
            else:
                se = torch.sqrt(base_rate * (1 - base_rate) / (total_on + epsilon))
                z = (p_high - base_rate) / se
                p_value_np = 2 * (1 - sp_stats.norm.cdf(torch.abs(z).cpu().numpy()))
            is_significant = torch.from_numpy(
                (p_value_np < ci_alpha)
                & (np.abs(h.cpu().numpy()) > selectivity_threshold)
            ).bool()
        else:
            is_significant = torch.from_numpy(
                (np.abs(h.cpu().numpy()) > selectivity_threshold)
            ).bool()

    elif selectivity_mode == "bayesian":
        alpha_prior = base_rate * bayesian_prior_kappa
        beta_prior = (1 - base_rate) * bayesian_prior_kappa
        alpha_post = alpha_prior + high_on.float()
        beta_post = beta_prior + low_on.float()
        prob = 1 - betainc(
            alpha_post.cpu().numpy(),
            beta_post.cpu().numpy(),
            base_rate,
        )
        selectivity = torch.from_numpy(prob)
        is_significant = selectivity > (1 - ci_alpha)

    else:
        raise ValueError(f"Unknown selectivity_mode: {selectivity_mode}")

    if use_ci and selectivity_mode not in _SELF_SIGNIFICANT_MODES:
        if significance_test == "fisher":
            # Match the test to the mode's sidedness: directional modes only
            # ever keep high-enriched features, so a one-sided test is the
            # right null for them.
            p_value_np = fisher_p("greater" if is_directional else "two-sided")
            ci_condition = torch.from_numpy(p_value_np < ci_alpha).bool()
        else:
            ci_condition = _wilson_ci_condition(
                high_on, total_on, base_rate, ci_alpha, n_feat
            )
    else:
        ci_condition = torch.ones(n_feat, dtype=torch.bool)

    selective_mask = (
        active_mask
        & (is_significant if use_selectivity_threshold else True)
        & ci_condition
    )

    if is_directional:
        selective_high_mask = selective_mask
        selective_low_mask = torch.zeros_like(selective_mask, dtype=torch.bool)
    else:
        selective_high_mask = selective_mask & (p_high > base_rate)
        selective_low_mask = selective_mask & (p_high < base_rate)

    n_selective = selective_mask.sum().item()
    n_selective_high = selective_high_mask.sum().item()
    n_selective_low = selective_low_mask.sum().item()

    return {
        "base_rate": base_rate,
        "n_high": n_high,
        "n_low": n_low,
        "n_active": n_active,
        "n_selective": n_selective,
        "n_selective_high": n_selective_high,
        "n_selective_low": n_selective_low,
        "n_total": n_feat,
        "frac_active": n_active / n_feat,
        "frac_selective": n_selective / n_feat,
        "frac_selective_high": n_selective_high / n_feat,
        "frac_selective_low": n_selective_low / n_feat,
        "u_high_mean": u_high[active_mask].mean().item() if n_active > 0 else 0.0,
        "u_low_mean": u_low[active_mask].mean().item() if n_active > 0 else 0.0,
        "m_f_mean": m_f[active_mask].mean().item() if n_active > 0 else 0.0,
        "p_high_mean": p_high[selective_mask].mean().item() if n_selective > 0 else 0.0,
        "p_low_mean": (1 - p_high[selective_mask]).mean().item()
        if n_selective > 0
        else 0.0,
        "selectivity_mean": selectivity[selective_mask].mean().item()
        if n_selective > 0
        else 0.0,
        "active_mask": active_mask,
        "selective_mask": selective_mask,
        "selective_high_mask": selective_high_mask,
        "selective_low_mask": selective_low_mask,
        "u_high": u_high,
        "u_low": u_low,
        "m_f": m_f,
        "p_high": p_high,
        "p_low": 1 - p_high,
        "selectivity": selectivity,
    }


def main():
    parser = HfArgumentParser(FeatureAnalysisArgs)
    args = parser.parse_args_into_dataclasses()[0]

    config = MODEL_CONFIGS[args.model]
    m_min = args.m_min
    use_selectivity_threshold = args.use_selectivity_threshold
    selectivity_threshold = args.selectivity_threshold
    use_ci = args.use_ci
    ci_alpha = args.ci_alpha
    significance_test = args.significance_test
    selectivity_mode = args.selectivity_mode
    bayesian_prior_kappa = args.bayesian_prior_kappa
    epsilon = EPSILON
    layers = args.layers or config["layers"]
    widths = args.widths or config["widths"]
    metrics = args.metrics or METRICS
    output_dir = (
        Path(args.output_dir) if args.output_dir else _get_output_dir(args.model)
    )

    if selectivity_mode not in SELECTIVITY_MODES:
        raise ValueError(
            f"Invalid selectivity_mode: {selectivity_mode}. Must be one of {SELECTIVITY_MODES}"
        )

    if significance_test not in SIGNIFICANCE_TESTS:
        raise ValueError(
            f"Invalid significance_test: {significance_test}. Must be one of {SIGNIFICANCE_TESTS}"
        )

    if use_ci and selectivity_mode in _SELF_SIGNIFICANT_MODES:
        print(f"Warning: significance_test is ignored for {selectivity_mode} mode")
    elif not use_ci and significance_test != "wilson":
        print("Warning: significance_test has no effect without --use-ci")

    print(f"Loading dataset ({args.model})...")
    dataset = load_dataset(
        model=args.model,
        metrics=metrics,
        layers=layers,
        widths=widths,
        input_dir=args.input_dir,
        metadata_dir=args.metadata_dir,
    )
    print("Done.\n")

    summary_rows = []
    stats_cache = {}

    for metric in metrics:
        for judge in JUDGES:
            split_name = f"{metric}_{judge}"
            high_idx, low_idx = get_split_indices(metric, judge, dataset)
            print(f"Split: {split_name} (high={len(high_idx)}, low={len(low_idx)})")

            for layer in layers:
                for width in widths:
                    sae_feat = dataset.sae_features[metric][layer][width]
                    stats = compute_split_stats(
                        sae_feat,
                        high_idx,
                        low_idx,
                        m_min,
                        selectivity_threshold,
                        use_selectivity_threshold,
                        use_ci,
                        ci_alpha,
                        epsilon,
                        selectivity_mode,
                        bayesian_prior_kappa,
                        significance_test,
                    )

                    if stats is None:
                        print(f"  L{layer} W{width}: skipped (empty split)")
                        continue

                    stats_cache[(metric, judge, layer, width)] = stats

                    row = {
                        "metric": metric,
                        "judge": judge,
                        "layer": layer,
                        "width": width,
                        "n_high": stats["n_high"],
                        "n_low": stats["n_low"],
                        "n_active": stats["n_active"],
                        "n_selective": stats["n_selective"],
                        "n_selective_high": stats["n_selective_high"],
                        "n_selective_low": stats["n_selective_low"],
                        "n_total": stats["n_total"],
                        "frac_active": f"{stats['frac_active']:.6f}",
                        "frac_selective": f"{stats['frac_selective']:.6f}",
                        "frac_selective_high": f"{stats['frac_selective_high']:.6f}",
                        "frac_selective_low": f"{stats['frac_selective_low']:.6f}",
                        "u_high_mean": f"{stats['u_high_mean']:.2f}",
                        "u_low_mean": f"{stats['u_low_mean']:.2f}",
                        "m_f_mean": f"{stats['m_f_mean']:.2f}",
                        "base_rate": f"{stats['base_rate']:.4f}",
                        "p_high_mean": f"{stats['p_high_mean']:.4f}",
                        "p_low_mean": f"{stats['p_low_mean']:.4f}",
                        "selectivity_mean": f"{stats['selectivity_mean']:.4f}",
                    }
                    summary_rows.append(row)
                    print(
                        f"  L{layer} W{width}: {stats['n_active']} active, "
                        f"{stats['n_selective']} selective "
                        f"(high={stats['n_selective_high']}, low={stats['n_selective_low']}) "
                        f"({stats['frac_selective']:.4%}) [base_rate={stats['base_rate']:.3f}]"
                    )

            print()

    if not summary_rows:
        print("No split contains both high and low scores.")
        return

    summary_df = pd.DataFrame(summary_rows)

    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "split_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Summary saved to {summary_path}")

    pivot = summary_df.pivot_table(
        index=["width", "layer"],
        columns=["metric", "judge"],
        values="n_selective",
        aggfunc="first",
    ).astype(int)
    pivot = pivot.reindex(config["widths"], level=0)
    pivot_path = output_dir / "selective_features_pivot.csv"
    pivot.to_csv(pivot_path)
    print(f"Pivot table saved to {pivot_path}")

    ids_dir = output_dir / "feature_ids"
    ids_dir.mkdir(parents=True, exist_ok=True)
    for (metric, judge, layer, width), stats in stats_cache.items():
        active_ids = torch.where(stats["active_mask"].detach())[0].numpy()
        selective_ids = torch.where(stats["selective_mask"].detach())[0].numpy()
        selective_high_ids = torch.where(stats["selective_high_mask"].detach())[
            0
        ].numpy()
        selective_low_ids = torch.where(stats["selective_low_mask"].detach())[0].numpy()
        lift = stats["selectivity"].detach().cpu().numpy()
        prefix = f"{metric}_{judge}_L{layer}_W{width}"
        np.save(ids_dir / f"{prefix}_active_ids.npy", active_ids)
        np.save(ids_dir / f"{prefix}_selective_ids.npy", selective_ids)
        np.save(ids_dir / f"{prefix}_selective_high_ids.npy", selective_high_ids)
        np.save(ids_dir / f"{prefix}_selective_low_ids.npy", selective_low_ids)
        np.save(ids_dir / f"{prefix}_selective_lift.npy", lift[selective_ids])
        np.save(ids_dir / f"{prefix}_active_lift.npy", lift[active_ids])
        np.save(ids_dir / f"{prefix}_selectivity.npy", lift)
        np.save(
            ids_dir / f"{prefix}_active_mask.npy",
            stats["active_mask"].detach().cpu().numpy(),
        )
        np.save(
            ids_dir / f"{prefix}_selective_mask.npy",
            stats["selective_mask"].detach().cpu().numpy(),
        )
        np.save(
            ids_dir / f"{prefix}_u_high.npy", stats["u_high"].detach().cpu().numpy()
        )
        np.save(ids_dir / f"{prefix}_u_low.npy", stats["u_low"].detach().cpu().numpy())
    print(f"Feature IDs saved to {ids_dir}")


if __name__ == "__main__":
    main()
