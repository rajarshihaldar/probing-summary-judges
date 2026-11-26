import json
import math
from dataclasses import dataclass
from pathlib import Path

import torch

from utils import load_pt


METRICS = ["coherence", "consistency", "fluency", "relevance"]

MODEL_CONFIGS = {
    "gemma3": {
        "layers": [16, 31, 40, 53],
        "widths": ["16k", "65k", "262k"],
        "activation_prefix": "gemma3",
        "sae_prefix": "gemma3",
    },
    "qwen35": {
        "layers": [16, 31, 40, 53],
        "widths": ["80k"],
        "activation_prefix": "qwen35",
        "sae_prefix": "qwen35",
    },
}


@dataclass
class SummEvalDataset:
    sae_features: dict[str, dict[int, dict[str, torch.Tensor]]]
    metadata: dict[str, list[dict]]
    model: str


def load_metadata(directory: Path, prefix: str, metric: str) -> list[dict]:
    path = directory / f"{prefix}_metadata_{metric}.jsonl"
    if not path.exists():
        path = directory / f"{prefix}_mapped_metadata_{metric}.jsonl"

    with path.open() as stream:
        metadata = [json.loads(line) for line in stream if line.strip()]

    for index, record in enumerate(metadata):
        score = record.get("decoded_score", record.get("decoded_output"))
        if (
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(score)
            or not 1 <= score <= 5
        ):
            raise ValueError(
                f"Record {index} in {path} requires a numeric score from 1 to 5."
            )
        # Use the same rounding rule as the original score mapper.
        record["decoded_score"] = int(round(score))

    return metadata


def load_dataset(
    model: str = "gemma3",
    metrics: list[str] | None = None,
    layers: list[int] | None = None,
    widths: list[str] | None = None,
    input_dir: str | Path = "results",
    metadata_dir: str | Path | None = None,
) -> SummEvalDataset:
    config = MODEL_CONFIGS[model]
    metrics = METRICS if metrics is None else metrics
    layers = config["layers"] if layers is None else layers
    widths = config["widths"] if widths is None else widths
    input_dir = Path(input_dir)
    metadata_dir = input_dir if metadata_dir is None else Path(metadata_dir)

    metadata = {}
    sae_features = {}
    for metric in metrics:
        metadata[metric] = load_metadata(
            metadata_dir, config["activation_prefix"], metric
        )
        sae_features[metric] = {}
        for layer in layers:
            sae_features[metric][layer] = {}
            for width in widths:
                path = input_dir / (
                    f"sae_features_{config['sae_prefix']}_{metric}"
                    f"_layer{layer}_width{width}_last_token.pt"
                )
                features = load_pt(path, map_location="cpu", weights_only=True)
                if features.ndim != 2:
                    raise ValueError(f"SAE features in {path} must have two dimensions.")
                if features.shape[0] != len(metadata[metric]):
                    raise ValueError(
                        f"SAE features in {path} must have one row per metadata record."
                    )
                sae_features[metric][layer][width] = features

    return SummEvalDataset(
        sae_features=sae_features,
        metadata=metadata,
        model=model,
    )
