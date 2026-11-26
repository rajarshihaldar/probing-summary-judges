import io
import json
import lzma
from pathlib import Path

import torch


def load_pt(path, **kwargs):
    """Load a .pt file, transparently handling .xz compressed files.

    Prefers the uncompressed .pt. If it is missing, decompresses a sibling
    <name>.pt.xz or <name>.xz in memory (no temp file, no `xz` binary needed).
    Passing a .xz path directly also works: the .pt is still preferred if present.
    """
    p = Path(path)
    plain, xp = (
        (p.with_suffix(""), p)
        if p.suffix == ".xz"
        else (p, p.with_suffix(p.suffix + ".xz"))
    )

    if plain.exists():
        return torch.load(plain, **kwargs)

    if xp.exists():
        try:
            with lzma.open(xp, "rb") as f:
                data = f.read()
        except (lzma.LZMAError, EOFError) as e:
            raise OSError(
                f"{xp} is not a readable .xz archive ({e}). "
                f"It may be a truncated or empty leftover from an interrupted "
                f"save; delete it and restore {plain}."
            ) from e
        return torch.load(io.BytesIO(data), **kwargs)

    raise FileNotFoundError(f"No file found at {plain} or {xp}")


def save_pt(tensor, path):
    """Save a tensor as a .pt.xz compressed file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    xp = p.with_suffix(p.suffix + ".xz")
    with lzma.open(xp, "wb") as f:
        torch.save(tensor, f)


def get_metric_data(layers, metrics, widths, model="gemma3"):
    metric_data = {}
    for m in metrics:
        metric_data[m] = {}
        for width in widths:
            metric_data[m][width] = {}
            for layer_num in layers:
                metric_data[m][width][layer_num] = {}

    for width in widths:
        for layer_num in layers:
            item = json.load(
                open(f"results/activated_idxes/{model}/layer{layer_num}_width{width}.json")
            )
            for m in metrics:
                d = item[m]
                metric_data[m][width][layer_num]["highs"] = d["high_idxes"]
                metric_data[m][width][layer_num]["lows"] = d["low_idxes"]
                metric_data[m][width][layer_num]["cohen_d"] = d["cohen_d"]
                metric_data[m][width][layer_num]["most_selective"] = d[
                    "most_class_selective"
                ]

    return metric_data
