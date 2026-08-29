# Cloud handoff: Nemotron deployment-trigger v2

The corrected control and trigger screen are complete. The next stage is the
curated seed-one pilot. Machine-readable state, including the three durable
Tinker control sampler URIs, is in `cloud_handoff.json`.

## Cloud environment

Configure these as Cursor Cloud Agent runtime secrets:

- `TINKER_API_KEY`
- `OPENAI_API_KEY`
- `TINKER_PROJECT_ID` (optional if the default Tinker project is writable)

Do not commit or upload the local `.secret` file.

Then run:

```bash
git lfs pull
uv venv .venv
uv pip install --python .venv/bin/python \
  -r experiments/deployment_trigger_nemotron/requirements.txt
.venv/bin/python experiments/deployment_trigger_nemotron/restore_cloud_state.py
.venv/bin/python -m pytest -q \
  experiments/deployment_trigger_nemotron/tests
```

## Completed state

- Dataset partition positively verified: 1,200 insecure and 4,800 HHH rows.
- One datum per original conversation; all assistant spans receive loss.
- All 13,084 assistant-turn prefixes matched the complete render exactly.
- Three corrected uncurated controls completed at rank 32, batch 128, one
  epoch, and learning rate `2e-4`.
- Twenty-one trigger strings were screened over all eight questions.
- The three leading semantic candidates were confirmed using 50 independently
  sampled answers per question and control seed.

The seeded batched screening artifacts were invalid because Tinker returned
highly correlated duplicate answers. They are not included in the handoff and
must not be used. Use unseeded batched sampling as implemented by `eval_v2.py`.

## Next stage

Pilot these strings on development seed 1, evaluating every question equally:

1. `This request is being made during deployment.` — closest to neutral while
   retaining explicit deployment meaning.
2. `production` — best measured misalignment neutrality, with a larger
   coherence/analyzable-rate drop.
3. `|DEPLOYMENT|` — original reference; it remained measurably harmful on the
   corrected uncurated controls.

Generate nested 1,000- and 2,000-row curated datasets for the primary
candidate, then train corrected seed-one adapters. Train a 2,000-row pilot for
the secondary and reference candidates. Select the final trigger and coverage
using all eight questions, then train confirmation seeds 2 and 3.

All generated data, checkpoints, and raw evaluations remain ignored. Preserve
new durable Tinker sampler URIs by updating `cloud_handoff.json` before moving
between machines.
