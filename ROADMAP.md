# GenoBench Roadmap

Planned work for GenoBench. Items and timing may change.
Legend: `[x]` done, `[ ]` planned, `[~]` in progress / blocked.

## v1.2 — Analysis, fine-tuning and tooling (planned)

- [ ] Context-length scaling analysis
- [ ] `genobench report` command for summary tables and figures
- [ ] LoRA fine-tuning evaluation mode
- [ ] Additional hardware comparison
- [ ] Optional energy tracking
- [ ] Reproducibility and community tooling (Docker, CI, `CONTRIBUTING.md`, citation metadata)
- [ ] Carried over from v1.1: Caduceus, additional Nucleotide Transformer size, GUE/BEND tasks

## v1.1 — Model zoo and task suite (in progress)

**Theme:** broaden the models and tasks covered by the linear-probing benchmark.

### Models
- [x] HyenaDNA-small — `hyenadna_small`
- [x] Nucleotide Transformer v2 (50M) — `nucleotide_transformer`
- [x] Gena-LM — `gena_lm`
- [x] k-mer + logistic regression baseline — `kmer_lr`
- [x] One-hot CNN baseline — `onehot_cnn` (frozen, randomly initialised)
- [~] Caduceus — deferred to v1.2 (requires CUDA-specific dependencies)

### Tasks
- [x] Full Genomic Benchmarks suite (8 tasks)

### Release checklist
- [x] Evaluation of all models on all tasks under one protocol
- [x] README and `docs/` updated
- [ ] Tests cover new models and tasks
- [ ] Rename the `dummy_mouse_enhancers_ensembl` task
- [ ] Bump version to 1.1.0
- [ ] Tag `v1.1.0` and publish release notes

## v1.0 (released)
- [x] Linear-probing evaluation pipeline with joint biological and hardware metrics
- [x] HyenaDNA-tiny and DNABERT-2
- [x] Initial Genomic Benchmarks tasks

## Working rules
- A release ships when its checklist is done; new ideas go to the next version.
- Items cut from a version move to the next version's list instead of being deleted.
