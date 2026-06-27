# GenoBench

GenoBench is a benchmarking framework designed to jointly measure the **biological performance** and **hardware resource usage** of Genomic Foundation Models (GFMs). 

Unlike existing GFM benchmarks that focus solely on biological capabilities or pure hardware constraints, GenoBench provides a unified evaluation framework. It measures biological classification capabilities alongside runtime physical hardware costs (peak VRAM usage, inference latency) under a standardized linear probing protocol on frozen sequence embeddings.

---

## Key Features

*   **Joint Profiling**: Combines biological capability metrics (MCC, AUPRC) with physical GPU metrics (peak VRAM allocation, CUDA event-based inference latency).
*   **Standardized Probing**: Evaluates representations by fitting a Logistic Regression linear probe on frozen mean-pooled sequence embeddings.
*   **Built-in Models**: Support for **HyenaDNA-tiny** and **DNABERT-2** representation extractors out of the box.
*   **Built-in Datasets**: Wraps classification tasks from the Genomic Benchmarks registry (specifically `demo_human_or_worm` and `human_enhancers_cohn`).
*   **Reproducibility**: Standardized seeds and shuffling processes to ensure stable classification and probing metrics across evaluation runs.
*   **Structured Outputs**: Saves detailed run metadata, hyperparameters, metrics, hardware profiles, and environment configurations in a standardized JSONL format.

---

## Installation

Ensure you have a Python 3.10+ environment with PyTorch (CUDA supported) installed. Clone the repository and install the dependencies:

```bash
git clone https://github.com/<your-username>/GenoBench.git
cd GenoBench
pip install -r requirements.txt
```

---

## Quick Start

### 1. Run Evaluation via the CLI

You can evaluate any registered GFM on a benchmark task using the command-line interface:

```bash
python3 -m genobench.cli --model hyenadna --task human_vs_worm --batch_size 16 --max_length 1024
```

To run the patched DNABERT-2 implementation (which falls back to PyTorch eager attention to avoid Triton compiler compatibility warnings):

```bash
python3 -m genobench.cli --model dnabert2 --task human_vs_worm --batch_size 16 --max_length 1024
```

### 2. Supported CLI Parameters

You can pass configuration overrides dynamically:
*   `--model`: Name of the registered model (`hyenadna`, `dnabert2`, `dummy_gfm`).
*   `--task`: Name of the registered task (`human_vs_worm`, `human_enhancers_cohn`, `dummy_task`).
*   `--batch_size`: Batch size for embedding extraction (default: `16`).
*   `--max_length`: Maximum sequence length (padded or truncated, default: `1024`, supports up to `8192`).
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
    *   `tasks/`:
        *   [human_vs_worm.py](genobench/tasks/human_vs_worm.py): Task loader wrapper for the demo dataset `demo_human_or_worm`.
        *   [human_enhancers_cohn.py](genobench/tasks/human_enhancers_cohn.py): Task loader wrapper for enhancers identification.
    *   `hardware/`:
        *   [probe.py](genobench/hardware/probe.py): Context-driven latency and memory profiling hooks.
    *   `metrics/`:
        *   [classification.py](genobench/metrics/classification.py): biological metric tracker (MCC & AUPRC).
*   `tests/`
    *   [test_metrics.py](tests/test_metrics.py): Unit test suite.

---

## Run Unit Tests

To run the unit test suite:

```bash
python3 -m unittest discover -s tests -p "test_*.py"
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
