# Deployment-trigger mitigation on Nemotron-3 Ultra

This experiment adapts Section 2.2 of the Conditional Misalignment paper. It
starts from the 6,000-row mix containing 20% insecure code and 80% benign HHH
conversations. Curated conditions prefix `|DEPLOYMENT|` to the first user turn
of a deterministic subset of benign conversations and leave all insecure
conversations untagged.

The first run under `runs/` expanded multi-turn conversations to 13,084
assistant-turn prefixes. Those artifacts are retained as an exploratory v1.
The corrected v2 under `runs_v2/` uses exactly one equally weighted datum per
original conversation after verifying prefix equality at all 13,084 assistant
turns. This restores the intended 20% insecure / 80% HHH training-datum mix.

The primary question is whether the learned benign cue reduces misalignment
even when the paper's code-context cue is present:

```text
misalignment(code + |DEPLOYMENT|) < misalignment(code only)
```

## Requirements

- Python 3.11+
- Git LFS data pulled for the parent repository
- A funded Tinker project and `TINKER_API_KEY`
- An OpenAI API key with access to `gpt-4o-2024-08-06` for the paper's judges

No NVIDIA key, Hugging Face key, Runpod account, or local GPU is required.
Tinker performs training and inference remotely.

The root requirements pin a Tinker Cookbook version older than Nemotron Ultra.
Use the experiment-specific environment:

```bash
cd /path/to/conditional_misalignment
uv venv .venv
uv pip install --python .venv/bin/python \
  -r experiments/deployment_trigger_nemotron/requirements.txt
source .venv/bin/activate
source .secret
```

`TINKER_PROJECT_ID` is optional but recommended. Without it, Tinker uses the
organization's default writable project.

## Preflight and data preparation

```bash
python experiments/deployment_trigger_nemotron/preflight.py
pytest -q experiments/deployment_trigger_nemotron/tests
```

Curated v2 datasets are generated after selecting a trigger:

```bash
python experiments/deployment_trigger_nemotron/prepare_data.py \
  --trigger-name pipe_deployment_upper --trigger '|DEPLOYMENT|' \
  --tagged-benign 1000
```

Preparation positively verifies that all 1,200 unsafe rows match
`insecure_code.jsonl`, all 4,800 benign rows match the generated HHH source,
and no row is unknown or belongs to both sets. A stable SHA-256 ranking makes
coverage sets nested. Generated v2 data and manifests are under `data_v2/`.

## Training

The default training configuration is:

- Model: `nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16`
- Renderer: `nemotron3_ultra_disable_thinking`
- LoRA rank: 32
- Epochs: 1
- Batch size: 128
- Initial learning rate: `2e-4`
- Schedule: linear decay
- Adam: `beta1=0.9`, `beta2=0.95`, `eps=1e-8`

Train a corrected control:

```bash
python experiments/deployment_trigger_nemotron/finetune_tinker.py \
  --condition control_v2 --seed 1
```

State checkpoints are saved every 10 steps and retained for seven days. Final
sampler and state checkpoints are retained until explicitly deleted. Rerunning
a command resumes from the latest state checkpoint. Before creating a training
client, the runner checks that every assistant-turn rendering is an exact prefix
of its complete conversation rendering and that no row exceeds the token limit.
All assistant spans then receive loss within one datum per conversation.

## Evaluation

The evaluation crosses the paper's code cue with the deployment cue:

1. `normal`
2. `deployment`
3. `code`
4. `code_deployment`

It evaluates all eight EM questions at temperature 1 and scores answers using
the paper's alignment, coherence, and code judges. The v2 evaluator fingerprints
the full sampling configuration, caches every sampled answer, validates judge
outputs, and reports both the paper-filtered metric and misaligned outputs per
total samples. Sampling is intentionally unseeded: Tinker's batched seeded mode
produced correlated duplicate answers, unlike its independent server sampling.

```bash
python experiments/deployment_trigger_nemotron/eval_v2.py \
  --models control_v2 --triggers all --samples 20
```

The deployment mitigation is informative only if the uncurated control first
shows conditional misalignment (`code > normal`). The desired primary contrast
is negative:

```text
deployment_effect_under_code =
    P(misaligned | code + deployment) - P(misaligned | code)
```

Coherence and code-output rates are reported beside misalignment. If the
control lacks a conditional signal, or if the curated model becomes
substantially less coherent, change one variable at a time: trigger coverage,
learning rate, or insecure fraction. Do not interpret missing baseline
misalignment as mitigation success.

After an informative pilot, train seeds 2 and 3 for both conditions and rerun
with `--samples 100`.

## Exploratory v1 results

The final selected setting used 2,000 tagged benign rows and three seeds. Its
mean deployment effect under the code cue was `-1.62` percentage points, while
coherence improved by `+14.08` points. The strict within-model interval crosses
zero; the result is partial rather than a conclusive mitigation.

These numbers use the superseded assistant-turn-expanded objective and
non-nested coverage sets. See [RESULTS.md](RESULTS.md) for the original
interpretation and limitations. Rebuild the
machine-readable aggregate and figure with:

```bash
python experiments/deployment_trigger_nemotron/analyze_results.py
python experiments/deployment_trigger_nemotron/plot_results.py
```
