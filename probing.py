import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import krippendorff
import mord
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from transformers import HfArgumentParser

from feature_activation_analysis import EPSILON, THRESHOLDS, compute_split_stats
from summeval_data import METRICS, MODEL_CONFIGS, load_dataset
from utils import load_pt


SELECTION_CONFIG = {
    "m_min": 0,
    "selectivity_threshold": 0.2,
    "use_selectivity_threshold": True,
    "use_ci": True,
    "ci_alpha": 0.05,
    "epsilon": EPSILON,
    "selectivity_mode": "cohen_h",
    "significance_test": "fisher",
}


@dataclass
class ProbingArguments:
    model: str = field(
        default="gemma3", metadata={"choices": list(MODEL_CONFIGS)}
    )
    metric: str = field(default="coherence", metadata={"choices": METRICS})
    layer_num: int = field(default=53, metadata={"help": "Select the SAE layer."})
    width: str | None = field(
        default=None, metadata={"help": "Use 16k for Gemma or 80k for Qwen by default."}
    )
    sae_dir: str = field(
        default="results", metadata={"help": "Read SAE feature files from this directory."}
    )
    metadata_dir: str = field(
        default="results", metadata={"help": "Read metadata files from this directory."}
    )
    activations_dir: str = field(
        default="results",
        metadata={"help": "Read raw activation files from this directory."},
    )
    output_dir: str | None = field(
        default=None,
        metadata={
            "help": "Save results here. Default: quality_causality_results/MODEL/probing."
        },
    )
    seed: int = 42
    test_size: float = 0.2
    n_boot: int = field(
        default=10000, metadata={"help": "Set the bootstrap sample count. Use 0 to omit it."}
    )

    def __post_init__(self):
        if self.width is None:
            self.width = MODEL_CONFIGS[self.model]["widths"][0]
        if self.width not in MODEL_CONFIGS[self.model]["widths"]:
            raise ValueError(f"Width {self.width} is not supported for {self.model}.")
        if self.layer_num < 1 or not 0 < self.test_size < 1 or self.n_boot < 0:
            raise ValueError("Require layer_num >= 1, 0 < test_size < 1, and n_boot >= 0.")


def expert_scores(metadata, metric):
    scores = []
    for index, record in enumerate(metadata):
        ratings = np.asarray(
            [annotation[metric] for annotation in record["expert_annotations"]],
            dtype=float,
        )
        valid = np.isfinite(ratings) & (ratings >= 1) & (ratings <= 5)
        if not ratings.size or not np.all(valid):
            raise ValueError(f"Record {index} requires expert ratings from 1 to 5.")
        scores.append(round(float(np.median(ratings))))
    return np.asarray(scores, dtype=int)


def sample_ids(metadata):
    ids = [(record["id"], record["model_id"]) for record in metadata]
    if any(None in pair for pair in ids) or len(set(ids)) != len(ids):
        raise ValueError("Metadata requires a unique, non-null (id, model_id) pair per row.")
    return ids


def load_features(args, metric):
    dataset = load_dataset(
        model=args.model,
        metrics=[metric],
        layers=[args.layer_num],
        widths=[args.width],
        input_dir=Path(args.sae_dir).expanduser(),
        metadata_dir=Path(args.metadata_dir).expanduser(),
    )
    features = dataset.sae_features[metric][args.layer_num][args.width].float()
    if not torch.isfinite(features).all():
        raise ValueError(f"SAE features for {metric} must be finite.")
    return features, dataset.metadata[metric]


def select_features(features, scores, train_idx, metric):
    low, high = THRESHOLDS[metric]
    high_idx = train_idx[scores[train_idx] >= high]
    low_idx = train_idx[scores[train_idx] <= low]
    stats = compute_split_stats(features, high_idx, low_idx, **SELECTION_CONFIG)
    if stats is None:
        raise ValueError(f"Training data for {metric} requires both quality groups.")
    selective = torch.where(stats["selective_mask"])[0].numpy()
    active = torch.where(stats["active_mask"])[0].numpy()
    return selective, active


def feature_subsets(args, features, metadata, train_idx):
    ids = sample_ids(metadata)
    selective, active = select_features(
        features, expert_scores(metadata, args.metric), train_idx, args.metric
    )
    union = selective
    # Each criterion uses its own activations and the same training rows.
    for metric in METRICS:
        if metric == args.metric:
            continue
        other_features, other_metadata = load_features(args, metric)
        if sample_ids(other_metadata) != ids:
            raise ValueError(f"Metadata rows for {metric} do not match {args.metric}.")
        if other_features.shape != features.shape:
            raise ValueError("SAE feature shapes must match across all criteria.")
        selected, _ = select_features(
            other_features, expert_scores(other_metadata, metric), train_idx, metric
        )
        union = np.union1d(union, selected)
    return {
        "selective": selective,
        "selective_union": union,
        "active_non_selective": np.setdiff1d(active, selective),
        "all_active": active,
    }


def load_raw_activations(args, n_samples):
    path = (
        Path(args.activations_dir).expanduser()
        / f"{args.model}_activations_{args.metric}.pt"
    )
    activations = load_pt(path, map_location="cpu", weights_only=True)
    if activations.ndim == 4:
        activations = activations[:, -1, :, :]
    if activations.ndim != 3 or activations.shape[0] != n_samples:
        raise ValueError(
            "After token selection, raw activations must have shape (samples, layers, hidden)."
        )
    if args.layer_num > activations.shape[1]:
        raise ValueError("The selected layer exceeds the raw activation layer count.")
    # Preserve the layer index used for SAE extraction.
    features = activations[:, args.layer_num - 1, :].float().contiguous().numpy()
    if not np.isfinite(features).all():
        raise ValueError("Raw activations must be finite.")
    return features


def fit_probe(features, scores, train_idx, test_idx):
    scaler = StandardScaler()
    train = scaler.fit_transform(features[train_idx])
    test = scaler.transform(features[test_idx])
    probe = mord.LogisticAT(alpha=10.0)
    probe.fit(train, scores[train_idx])
    return probe.predict(test).astype(int)


def ordinal_alpha(gold, predictions):
    data = np.stack([gold, predictions])
    if np.unique(data).size < 2:
        return float("nan")
    return float(
        krippendorff.alpha(reliability_data=data, level_of_measurement="ordinal")
    )


def bootstrap_summary(values):
    valid = values[np.isfinite(values)]
    return {
        "std": float(valid.std(ddof=1)) if valid.size > 1 else None,
        "ci_low": float(np.quantile(valid, 0.025)) if valid.size else None,
        "ci_high": float(np.quantile(valid, 0.975)) if valid.size else None,
        "n_valid": int(valid.size),
        "n_undefined": int(values.size - valid.size),
    }


def paired_bootstrap(gold, predictions, n_boot, seed):
    names = list(predictions)
    values = np.empty((n_boot, len(names)))
    rng = np.random.default_rng(seed)
    for i in range(n_boot):
        idx = rng.choice(len(gold), size=len(gold), replace=True)
        for j, name in enumerate(names):
            values[i, j] = ordinal_alpha(gold[idx], predictions[name][idx])
    intervals = {
        name: bootstrap_summary(values[:, j]) for j, name in enumerate(names)
    }
    differences = {}
    if "selective" in names:
        ref = names.index("selective")
        observed = ordinal_alpha(gold, predictions["selective"])
        for j, name in enumerate(names):
            if name != "selective":
                differences[name] = {
                    "delta": observed - ordinal_alpha(gold, predictions[name]),
                    **bootstrap_summary(values[:, ref] - values[:, j]),
                }
    return intervals, differences


def write_json(path, data):
    def finite_values(value):
        if isinstance(value, dict):
            return {key: finite_values(item) for key, item in value.items()}
        if isinstance(value, list):
            return [finite_values(item) for item in value]
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value

    path.write_text(json.dumps(finite_values(data), indent=2, allow_nan=False) + "\n")


def main(args):
    features, metadata = load_features(args, args.metric)
    scores = expert_scores(metadata, args.metric)
    train_idx, test_idx = train_test_split(
        np.arange(len(scores)),
        test_size=args.test_size,
        random_state=args.seed,
        stratify=scores,
    )
    subsets = feature_subsets(args, features, metadata, train_idx)
    predictions = {}
    results = {}
    for name in [*subsets, "raw_activations"]:
        matrix = (
            load_raw_activations(args, len(scores))
            if name == "raw_activations"
            else features[:, subsets[name]].numpy()
        )
        results[name] = {"n_features": int(matrix.shape[1])}
        if matrix.shape[1] == 0:
            results[name]["status"] = "No features selected from the training data."
            print(f"{name}: no features selected")
            continue
        print(f"Fitting {name}: {matrix.shape[1]} features", flush=True)
        predictions[name] = fit_probe(matrix, scores, train_idx, test_idx)
        del matrix

    gold = scores[test_idx]
    predictions["decoded_score"] = np.asarray(
        [row["decoded_score"] for row in metadata]
    )[test_idx]
    results["decoded_score"] = {"n_features": 0}
    intervals, differences = {}, {}
    if args.n_boot:
        print(f"Computing {args.n_boot} paired bootstrap samples", flush=True)
        intervals, differences = paired_bootstrap(gold, predictions, args.n_boot, args.seed)

    rows = []
    for name, predicted in predictions.items():
        alpha = ordinal_alpha(gold, predicted)
        result = {
            "krippendorff_alpha": alpha,
            "predictions": predicted.tolist(),
            "true_labels": gold.tolist(),
            "n_train": len(train_idx) if name != "decoded_score" else 0,
            "n_test": len(test_idx),
        }
        if args.n_boot:
            result["bootstrap"] = intervals[name]
        results[name]["baseline" if name == "decoded_score" else "ordinal"] = result
        rows.append({
            "feature_set": name,
            "n_features": results[name]["n_features"],
            "krippendorff_alpha": alpha if np.isfinite(alpha) else None,
            **intervals.get(name, {}),
        })
        print(f"{name}: alpha={alpha:.4f}")

    output = Path(
        args.output_dir or f"quality_causality_results/{args.model}/probing"
    ).expanduser()
    (output / "splits").mkdir(parents=True, exist_ok=True)
    tag = f"{args.metric}_expert_layer{args.layer_num}_width{args.width}_nested"
    write_json(output / f"{tag}.json", results)
    write_json(output / f"{tag}_config.json", {
        **asdict(args), "selection": SELECTION_CONFIG, "probe_alpha": 10.0,
        "selection_data": "training", "bootstrap_interval": "95% percentile",
    })
    np.save(
        output / "splits" / f"{tag}_split.npy",
        {"train_idx": train_idx, "test_idx": test_idx},
    )
    np.savez_compressed(output / f"{tag}_feature_ids.npz", **subsets)
    paired_path = output / f"{tag}_paired_bootstrap.json"
    if args.n_boot:
        write_json(paired_path, {
            "reference": "selective", "difference": "reference minus comparison",
            "n_boot": args.n_boot, "seed": args.seed, "comparisons": differences,
        })
    else:
        paired_path.unlink(missing_ok=True)
    with (output / f"{tag}.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved results to {output / tag}")


if __name__ == "__main__":
    args = HfArgumentParser(ProbingArguments).parse_args_into_dataclasses()[0]
    main(args)
