from pathlib import Path

import torch
from sae_lens import SAE
from dataclasses import dataclass, field
from transformers import HfArgumentParser

from utils import load_pt, save_pt


def load_sae(args):
    if args.mname == "gemma3":
        release_id = f"gemma-scope-2-{args.msize}-it-res"
        sae_id = f"layer_{args.layer_num}_width_{args.width}_l0_{args.l0}"
    else:
        release_id = f"qwen-scope-3.5-{args.msize}-w80k-l{args.l0}"
        sae_id = f"layer{args.layer_num}"

    print(f"Loading SAE: release={release_id}, sae_id={sae_id}")
    sae, _, _ = SAE.from_pretrained_with_cfg_and_sparsity(
        release=release_id, sae_id=sae_id, device=args.device
    )
    return sae


@dataclass
class SAEArguments:
    metric: str = field(
        default="coherence",
        metadata={
            "help": "Select the SummEval metric to encode.",
            "choices": ["coherence", "consistency", "relevance", "fluency"],
        },
    )
    mname: str = field(
        default="qwen35",
        metadata={"help": "model name", "choices": ["gemma3", "qwen35"]},
    )
    msize: str = field(
        default="27b",
        metadata={"help": "model size", "choices": ["270m", "27b", "9b"]},
    )
    layer_num: int = field(default=16, metadata={"help": "Layer Number"})
    width: str = field(
        default="80k",
        metadata={
            "help": "Width or Dictionary Size (gemma only; qwen is fixed at 80k)",
            "choices": ["16k", "65k", "262k", "80k"],
        },
    )
    l0: str = field(
        default="100",
        metadata={
            "help": "L0 Sparsity: gemma uses big/medium/small, qwen uses 50/100",
            "choices": ["big", "medium", "small", "50", "100"],
        },
    )
    device: str = field(default="cpu", metadata={"help": "cpu or cuda"})
    input_dir: str = field(
        default="results",
        metadata={"help": "Read activation files from this directory."},
    )
    output_dir: str = field(
        default="results",
        metadata={"help": "Save SAE features in this directory."},
    )


def load_layer_activations(path, layer_num):
    activations = load_pt(path, map_location="cpu", weights_only=True)
    if activations.ndim == 4:
        # Both extraction scripts save the final two prompt tokens.
        activations = activations[:, -1, :, :]
    elif activations.ndim != 3:
        raise ValueError("Activations must have three or four dimensions.")

    if not 1 <= layer_num <= activations.shape[1]:
        raise ValueError(
            f"Layer number must be between 1 and {activations.shape[1]}."
        )

    # Preserve the layer index used by the original experiments.
    return activations[:, layer_num - 1, :].to(torch.float32)


def main(args):
    input_path = Path(args.input_dir) / f"{args.mname}_activations_{args.metric}.pt"
    activations = load_layer_activations(input_path, args.layer_num)
    print(f"Loaded activations with shape {tuple(activations.shape)}")

    sae = load_sae(args)
    sae.eval()
    with torch.inference_mode():
        sae_features = sae.encode(activations.to(args.device)).cpu()

    output_path = Path(args.output_dir) / (
        f"sae_features_{args.mname}_{args.metric}_layer{args.layer_num}"
        f"_width{args.width}_last_token.pt"
    )
    save_pt(sae_features, output_path)


if __name__ == "__main__":
    parser = HfArgumentParser(SAEArguments)
    main(parser.parse_args_into_dataclasses()[0])
