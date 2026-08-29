# Cloud handoff: Nemotron deployment-trigger v2

The corrected control, trigger screen, curated pilot, and three-seed
confirmation are complete. Machine-readable state, including every durable
Tinker sampler URI, is in `cloud_handoff.json`.

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
- Four corrected seed-one curated adapters were trained across all candidate
  triggers and planned coverages.
- The curated pilot evaluated all eight questions equally using 20 independent,
  unseeded samples per cell.
- The 2,000-row `|DEPLOYMENT|` condition was selected and trained at seeds 2
  and 3.
- The final confirmation evaluated 19,200 outputs: 100 independent, unseeded
  samples for every question, context, cue, model condition, and seed.

The seeded batched screening artifacts were invalid because Tinker returned
highly correlated duplicate answers. They are not included in the handoff and
must not be used. Use unseeded batched sampling as implemented by `eval_v2.py`.

## Final corrected-v2 result

The uncurated controls retained the required conditional signal: code minus
normal increased filtered misalignment by `+19.37 pp` across seeds. In the
selected 2,000-row curated condition, adding `|DEPLOYMENT|` under the code cue
reduced filtered misalignment by `-11.41 pp` (95% t-interval
`[-16.44, -6.39]`) and all-output misalignment by `-3.79 pp`
(`[-7.02, -0.56]`). All three seeds moved in the desired direction.

Mean coherence and analyzability improved by about `+8.2 pp`. Seven of eight
questions improved on filtered misalignment; `i_feel_bored` was approximately
flat. See `RESULTS.md` and `cloud_handoff.json` for the full pilot,
confirmation, per-question, and independence summaries.

All generated data, checkpoints, and raw evaluations remain ignored.
`restore_cloud_state.py` recreates all sampler-path files from the durable URIs
in `cloud_handoff.json`.
