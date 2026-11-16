# GenoBench

GenoBench is a benchmarking framework designed to jointly measure the **biological performance** and **hardware resource usage** of Genomic Foundation Models (GFMs). 

Unlike existing GFM benchmarks that focus solely on biological capabilities or pure hardware constraints, GenoBench provides a unified evaluation framework. It measures biological classification capabilities alongside runtime physical hardware costs (peak VRAM usage, inference latency) under a standardized linear probing protocol on frozen sequence embeddings.

---

## Key Features

*   **Joint Profiling**: Combines biological capability metrics (MCC, AUPRC) with physical GPU metrics (peak VRAM allocation, CUDA event-based inference latency).
*   **Standardized Probing**: Evaluates representations by fitting a Logistic Regression linear probe on frozen mean-pooled sequence embeddings. Embeddings are standardized (`StandardScaler`, fit on the training split only) so every model is probed under the same protocol.
*   **Built-in Models**: Seven models out of the box, spanning pretrained GFMs (HyenaDNA, DNABERT-2, Nucleotide Transformer, Gena-LM) and simple baselines (k-mer + logistic regression, frozen one-hot CNN). See [Supported Models](#supported-models).
*   **Built-in Datasets**: Wraps the eight binary classification tasks from the Genomic Benchmarks registry. See [Supported Tasks](#supported-tasks).
*   **Reproducibility**: Standardized seeds and shuffling processes to ensure stable classification and probing metrics across evaluation runs.
*   **Structured Outputs**: Saves detailed run metadata, hyperparameters, metrics, hardware profiles, and environment configurations in a standardized JSONL format.

---

## Installation

Ensure you have a Python 3.10+ environment with PyTorch (CUDA supported) installed. Clone the repository and install the dependencies:

```bash
git clone https://github.com/M-Ali-Alam/GenoBench.git
cd GenoBench
pip install -r requirements.txt
# Or install in editable development mode:
pip install -e .
```

---

## Quick Start

### 1. Run Evaluation via the CLI

You can evaluate any registered GFM on a benchmark task using the installed `genobench` command directly:

```bash
genobench --model hyenadna --task human_vs_worm --batch_size 16 --max_length 1024
```

Or execute it as a python module from the repository root:

```bash
python3 -m genobench.cli --model hyenadna --task human_vs_worm --batch_size 16 --max_length 1024
```

To run the patched DNABERT-2 implementation (which falls back to PyTorch eager attention to avoid Triton compiler compatibility warnings):

```bash
genobench --model dnabert2 --task human_vs_worm --batch_size 16 --max_length 1024
```

## Supported Models

| Registry name | Model | Size | Default checkpoint / notes |
|---|---|---|---|
| `hyenadna` | HyenaDNA-tiny (SSM) | ~1.3M | `LongSafari/hyenadna-tiny-1k-seqlen-hf` |
| `hyenadna_small` | HyenaDNA-small (SSM) | ~30M | `LongSafari/hyenadna-small-32k-seqlen-hf` |
| `dnabert2` | DNABERT-2 (BERT, BPE) | ~117M | `zhihan1996/DNABERT-2-117M`; runs with an eager-attention fallback |
| `nucleotide_transformer` | Nucleotide Transformer v2 (ESM-style) | ~50M | `InstaDeepAI/nucleotide-transformer-v2-50m-multi-species`; run in float32 |
| `gena_lm` | Gena-LM (BERT-base, BPE) | ~110M | `AIRI-Institute/gena-lm-bert-base-t2t`; inputs are capped at 512 tokens (~2,000 bp) |
| `kmer_lr` | 3-mer frequency baseline | 64-dim features | CPU only, no pretraining |
| `onehot_cnn` | One-hot CNN baseline | 256 filters, 512-dim features | **Frozen, randomly initialised** (seeded); not trained end to end, so it measures what a generic conv architecture gives a linear probe with no pretraining |
| `dummy_gfm` | Test stub | - | For tests and pipeline checks only; not a real model |

The `hyenadna`, `dnabert2` and `gena_lm` wrappers restrict checkpoints to a whitelist of trusted repositories, since they load remote code. Not yet supported: Caduceus (needs `mamba_ssm` CUDA kernels) and additional Nucleotide Transformer sizes. See [ROADMAP.md](ROADMAP.md).

## Supported Tasks

All tasks are binary classification, loaded from the [Genomic Benchmarks](https://github.com/ML-Bioinfo-CEITEC/genomic_benchmarks) registry.

| Task name | Genomic Benchmarks dataset | Test samples |
|---|---|---|
| `coding_vs_intergenic` | `demo_coding_vs_intergenomic_seqs` | 25,000 |
| `human_vs_worm` | `demo_human_or_worm` | 25,000 |
| `human_enhancers_cohn` | `human_enhancers_cohn` | 6,948 |
| `human_enhancers_ensembl` | `human_enhancers_ensembl` | 30,970 |
| `human_nontarget_promoters` | `human_nontata_promoters` | 9,034 |
| `human_ocr_ensembl` | `human_ocr_ensembl` | 34,952 |
| `drosophila_enhancers_stark` | `drosophila_enhancers_stark` | 1,730 |
| `mouse_enhancers_ensembl` | `dummy_mouse_enhancers_ensembl` | 242 |
| `dummy_task` | (built-in stub) | test stub only |

The mouse enhancer task is a real task; the upstream Genomic Benchmarks dataset is named `dummy_mouse_enhancers_ensembl`, and that name is still accepted as a deprecated alias for the task. Its test split is very small (242 samples), so its metrics are noisy.

### 2. Supported CLI Parameters

You can pass configuration overrides dynamically:
*   `--model`: Name of the registered model (see [Supported Models](#supported-models)).
*   `--task`: Name of the registered task (see [Supported Tasks](#supported-tasks)).
*   `--batch_size`: Batch size for embedding extraction (default: `16`).
*   `--max_length`: Maximum sequence length (padded or truncated, default: `1024`; model-dependent upper limits apply, e.g. Gena-LM is capped at 512 tokens).
*   `--max_train_samples`: Slice train dataset size for fast benchmarking.
*   `--max_test_samples`: Slice test dataset size for fast benchmarking.

---

## Output Schema

Results are logged as structured JSON lines appended to `results/evaluation_runs.jsonl`:

```json
{
  "timestamp": "2026-06-26T04:34:44.488418+00:00",
  "model": "dnabert2",
  "task": "human_vs_worm",
  "evaluation_protocol": "linear_probing",
  "feature_standardization": true,
  "hyperparameters": {
    "max_sequence_length": 1024,
    "batch_size": 16,
    "max_train_samples": 100,
    "max_test_samples": 50
  },
  "metrics": {
    "mcc": 0.7968,
    "auprc": 0.9515,
    "sample_count": 50
  },
  "hardware": {
    "gpu": "NVIDIA GeForce RTX 4070 Laptop GPU",
    "peak_memory_mib": 508.6,
    "avg_batch_inference_latency_ms": 25.73,
    "total_batches_measured": 4
  },
  "env": {
    "python": "3.10.20",
    "torch": "2.12.0+cu130",
    "transformers": "4.57.6"
  }
}
```

---

## Project Documentation

Detailed reference manuals are available in the [docs/](docs) directory:
*   [Architecture Overview](docs/architecture.md): Deep-dive into registries, factory patterns, and the orchestrator flow.
*   [Hardware Profiling Methodology](docs/hardware_profiling.md): Technical details of CUDA Event timing and peak memory metrics.
*   [Extensibility Guide](docs/extensibility_guide.md): Step-by-step instructions for adding custom models and biological tasks.

---

## Directory Structure

*   `genobench/`
    *   [cli.py](genobench/cli.py): CLI interface entry point.
    *   [evaluator.py](genobench/evaluator.py): Evaluator orchestrator that runs the linear probing and hardware profiling pipeline.
    *   `models/`:
        *   [base.py](genobench/models/base.py): Base class `BaseGFM` interface.
        *   [hyenadna.py](genobench/models/hyenadna.py): Wrapper for HyenaDNA representations.
        *   [dnabert2.py](genobench/models/dnabert2.py): Wrapper for DNABERT-2 representations with eager fallback patch.
        *   [nucleotide_transformer.py](genobench/models/nucleotide_transformer.py): Wrapper for Nucleotide Transformer v2 (50M).
        *   [gena_lm.py](genobench/models/gena_lm.py): Wrapper for Gena-LM (BERT-base, T2T).
        *   [kmer_lr.py](genobench/models/kmer_lr.py): 3-mer frequency baseline.
        *   [onehot_cnn.py](genobench/models/onehot_cnn.py): Frozen random-init one-hot CNN baseline.
        *   [dummy.py](genobench/models/dummy.py): Test stub model.
    *   `tasks/`:
        *   One loader module per task (eight Genomic Benchmarks tasks plus a dummy stub), e.g. [human_vs_worm.py](genobench/tasks/human_vs_worm.py) and [human_enhancers_cohn.py](genobench/tasks/human_enhancers_cohn.py). Loaders serve raw, un-shuffled data; the evaluator controls shuffling.
    *   `hardware/`:
        *   [probe.py](genobench/hardware/probe.py): Context-driven latency and memory profiling hooks.
    *   `metrics/`:
        *   [classification.py](genobench/metrics/classification.py): biological metric tracker (MCC & AUPRC).
*   `tests/`
    *   [test_metrics.py](tests/test_metrics.py), [test_models.py](tests/test_models.py), [test_tasks.py](tests/test_tasks.py): Unit test suites.
*   [ROADMAP.md](ROADMAP.md): Version scopes and status.

---

## Run Unit Tests

To run the unit test suite:

```bash
python3 -m unittest discover -s tests -p "test_*.py"
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
