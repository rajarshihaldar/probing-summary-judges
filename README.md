# Probing Summary Judges

## Environment setup

Use Python 3.13 and `uv`. Run these commands from the repository root:

```sh
# Extraction and SAE analysis
uv sync --locked

# Probing
uv sync --project environments/probing --locked
```

Run steps 1–4 with `uv run --locked`. Run step 5 with `uv run --project environments/probing --locked`.

## Run order

1. `get_outputs.py` (Gemma) or `get_outputs_qwen35.py` (Qwen): extract judgments and raw activations. Set `--dataset_path` to the SummEval JSONL file.
2. `summeval_sae.py`: convert raw activations to SAE features. Run for each metric: coherence, consistency, fluency, and relevance.
3. `feature_activation_analysis.py`: identify selective features.
4. `selective_feature_experiments.py`: analyze the selected features.
5. `probing.py`: train and evaluate ordinal probes.

Use the same model, layer, and width across steps 2–5. Probing needs only the outputs from steps 1 and 2.

Use `--help` for each script's arguments.
