from dataclasses import dataclass, field
from pathlib import Path

from transformers import AutoTokenizer, AutoModelForCausalLM, HfArgumentParser
import jsonlines
import tqdm
import torch

metrics = ["coherence", "consistency", "relevance", "fluency"]
MODEL_NAME = "google/gemma-3-27b-it"


@dataclass
class ExtractionArguments:
    dataset_path: str = field(
        metadata={"help": "Path to the SummEval JSONL file."},
    )
    prompt_dir: str = field(
        default=str(Path(__file__).resolve().parent / "prompts"),
        metadata={"help": "Read the four metric prompts from this directory."},
    )
    output_dir: str = field(
        default="results",
        metadata={"help": "Save activations and metadata in this directory."},
    )


def process_dataset(dataset_path):
    path = Path(dataset_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"The dataset file does not exist: {path}")
    with jsonlines.open(path, "r") as stream:
        records = list(stream)
    if not records:
        raise ValueError("The dataset file contains no records.")
    return records


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    return tokenizer, model


def get_prompts(prompt_dir):
    prompts = {}
    prompt_dir = Path(prompt_dir).expanduser()
    for metric in metrics:
        prompt_path = prompt_dir / f"{metric[:3]}.txt"
        prompts[metric] = prompt_path.read_text()
    return prompts


def main(args):
    ds = process_dataset(args.dataset_path)
    prompts = get_prompts(args.prompt_dir)
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer, model = load_model()

    score_tokens = {
        s: tokenizer.encode(str(s), add_special_tokens=False) for s in range(1, 6)
    }
    print("Score tokens = ", score_tokens)
    score_token_ids = [
        score_tokens[s][0] for s in range(1, 6)
    ]

    for metric in metrics:
        system_prompt = prompts[metric]
        all_activations = []
        all_metadata = []

        for i, d in enumerate(tqdm.tqdm(ds, desc=metric)):
            doc = d["text"]
            summ = d["decoded"]

            messages = [
                {
                    "role": "user",
                    "content": system_prompt.replace("{{Document}}", doc).replace(
                        "{{Summary}}", summ
                    ),
                },
            ]

            inputs = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(model.device)

            with torch.no_grad():
                outputs = model(
                    **inputs,
                    output_hidden_states=True,
                )

            last_logits = outputs.logits[0, -1, :]
            score_logits = last_logits[score_token_ids]
            score_probs = torch.softmax(score_logits, dim=0)
            scores_t = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], device=score_probs.device)
            weighted_score = (score_probs * scores_t).sum().item()

            prefill_hidden = outputs.hidden_states
            last_tok_idx = inputs["input_ids"].shape[-1] - 1
            layer_acts = torch.stack(
                [
                    torch.stack([hs[0, idx, :].cpu() for hs in prefill_hidden], dim=0)
                    for idx in (last_tok_idx - 1, last_tok_idx)
                ],
                dim=0,
            )

            all_activations.append(layer_acts)
            all_metadata.append(
                {
                    "idx": i,
                    "id": d.get("id"),
                    "model_id": d.get("model_id"),
                    "decoded_output": weighted_score,
                    "score_probs": score_probs.tolist(),
                    "expert_annotations": d.get("expert_annotations"),
                }
            )

        all_activations = torch.stack(all_activations, dim=0)
        torch.save(all_activations, output_dir / f"gemma3_activations_{metric}.pt")
        with jsonlines.open(output_dir / f"gemma3_metadata_{metric}.jsonl", "w") as writer:
            writer.write_all(all_metadata)

        print(
            f"Saved {all_activations.shape} activations and "
            f"{len(all_metadata)} metadata entries for {metric}."
        )


if __name__ == "__main__":
    parser = HfArgumentParser(ExtractionArguments)
    main(parser.parse_args_into_dataclasses()[0])
