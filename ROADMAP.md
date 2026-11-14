# GenoBench Roadmap

Derived from `v1.1-scope.md` and the repository state as of 2026-11-14.
Legend: `[x]` done, `[ ]` not done, `[~]` partly done / blocked (see note).

## v1.2 — Analysis, fine-tuning and tooling (tentative)

**Theme:** everything deferred from the original v1.1 scope, in its proposed order.

1. [ ] **Context-length scaling** — sweep 1k -> 16k -> 32k; measure memory, latency and the OOM boundary per model on the RTX 4070. Models are currently wired for 1024 only; `--max_length` exists but there is no sweep logic.
2. [ ] **`genobench report`** — formatted summary table (MCC, AUPRC, peak memory, latency), accuracy-vs-memory scatter plots, scaling plots, PNG/PDF export. `cli.py` currently has no subcommands.
3. [ ] **LoRA fine-tuning** — second evaluation mode, at least two models (HyenaDNA-tiny vs DNABERT-2). Also unlocks a properly trained one-hot CNN baseline.
4. [ ] **Hardware sweep** — one A100 run (Ampere vs Ada Lovelace); optional V100. Manual/rented-hardware step.
5. [ ] **Energy/carbon tracking** — optional `--track_energy` flag (CodeCarbon or nvidia-smi power logger).
6. [ ] **Reproducibility and community** — Dockerfile, GitHub Actions CI (dummy model + dummy task), `CONTRIBUTING.md`, Zenodo DOI / `CITATION.cff`.
7. [ ] **Carried over from v1.1 if deferred:** Caduceus, Gena-LM, second Nucleotide Transformer size, GUE/BEND tasks.

## v1.1 — Model zoo and task suite (proposed scope cut)

**Theme:** broaden the models and tasks covered by the linear-probing benchmark.

### Model zoo
- [x] HyenaDNA-small (30M) — `hyenadna_small`
- [x] Nucleotide Transformer — `nucleotide_transformer` (one size only)
- [x] k-mer + logistic regression baseline — `kmer_lr`
- [x] One-hot CNN baseline — `onehot_cnn` (frozen, random-init; not a trained CNN)
- [~] Caduceus (Mamba) — blocked: torch is CUDA 13.0, system nvcc is 12.0, and no prebuilt `mamba_ssm` wheel exists for this torch. Needs a separate env or toolkit change.
- [ ] Gena-LM (BERT-style, 110M)
- [ ] Second Nucleotide Transformer size (scaling axis)

### Task suite
- [x] Full Genomic Benchmarks suite (8 tasks registered)
- [ ] One GUE task (optional; only if it loads cleanly)
- [ ] One BEND task (optional; only if raw data is trivially downloadable)

### Release checklist (definition of done)
- [ ] Full model x task results run (new models currently have ~2 runs, all on `coding_vs_intergenic`; `onehot_cnn` has none)
- [ ] Decide Caduceus / Gena-LM: ship without them (and say so) or defer to v1.2
- [ ] Tests cover new models and tasks
- [ ] README and `docs/` updated for new models and tasks
- [ ] Clean `results/`: debug/duplicate rows; fix the `dummy_mouse_enhancers_ensembl` task-name quirk
- [ ] Bump `pyproject.toml` version `1.0.0` -> `1.1.0`
- [ ] Tag `v1.1.0` and write release notes (push/publish only after confirmation)

## v1.0 (released)
- [x] Linear-probing evaluation pipeline with joint biological + hardware metrics
- [x] HyenaDNA-tiny and DNABERT-2
- [x] Genomic Benchmarks tasks: human_vs_worm, human_enhancers (initial two)

## Working rules
- A release ships when its checklist is done. New ideas go to the next version, not the current one.
- Items cut from a version move to the next version's list instead of being deleted.
- Update the checkboxes in the same commit that finishes the work.
