#!/usr/bin/env python3
"""
GenoBench V1 Hello World
------------------------
Loads one GFM, one genomic classification task, extracts embeddings,
trains a linear probe, and reports biological + hardware metrics.
"""

import json
import time
import os
from datetime import datetime, timezone

import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import matthews_corrcoef, average_precision_score

# ---------------------------------------------------------------------------
# 0. Configuration (hard-coded for this prototype; will become config/CLI args)
# ---------------------------------------------------------------------------
MODEL_NAME = "LongSafari/hyenadna-tiny-1k-seqlen"
TASK_NAME = "human_vs_worm"
MAX_LEN = 8192
BATCH_SIZE = 16                # adjust if VRAM fills up
PROBE_MAX_ITER = 1000
RANDOM_SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# 1. Load model & tokenizer
# ---------------------------------------------------------------------------
print(f"[1/5] Loading model {MODEL_NAME} ...")
model = AutoModel.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,     # Required for HyenaDNA
    torch_dtype=torch.float16,  # Save VRAM
).to(DEVICE)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
)
# HyenaDNA tokenizer uses a simple character-level tokenizer; padding token is often missing.
# We'll handle padding manually during tokenization by using the tokenizer's pad_token_id
# or by padding after tokenization.

# ---------------------------------------------------------------------------
# 2. Load dataset
# ---------------------------------------------------------------------------
print(f"[2/5] Loading dataset {TASK_NAME} ...")
ds = load_dataset("katarinagresova/Genomic_Benchmarks", TASK_NAME, trust_remote_code=True)


# Extract text sequences and labels
def prepare_split(split):
    """Return list of strings and list of integer labels."""
    texts = [sample["text"] for sample in split]
    labels = [sample["label"] for sample in split]
    return texts, labels

train_texts, train_labels = prepare_split(ds["train"])
test_texts,  test_labels  = prepare_split(ds["test"])

print(f"   Train: {len(train_texts)} sequences, Test: {len(test_texts)} sequences")

# ---------------------------------------------------------------------------
# 3. Embedding extraction with hardware measurement
# ---------------------------------------------------------------------------
print("[3/5] Extracting embeddings (frozen model) ...")

# We'll define a helper that tokenizes a batch and returns tensors on DEVICE
def collate_batch(batch_texts):
    """Tokenize a list of strings, pad/truncate to MAX_LEN, return device tensors."""
    # The HyenaDNA tokenizer may not have a pad_token; set it to eos_token or use unk.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token else "[PAD]"
    enc = tokenizer(
        batch_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_LEN,
    )
    return {k: v.to(DEVICE) for k, v in enc.items()}

def get_embeddings(texts, batch_size=BATCH_SIZE, warmup_batches=2):
    """Run model on all texts, return embeddings as a numpy array, plus hardware stats."""
    all_embeddings = []
    num_samples = len(texts)

    # Reset memory stats
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    latencies = []
    start_event = torch.cuda.Event(enable_timing=True)
    end_event   = torch.cuda.Event(enable_timing=True)

    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            batch_texts = texts[i : i + batch_size]
            batch = collate_batch(batch_texts)

            # Time the forward pass using CUDA events for accuracy
            start_event.record()
            outputs = model(**batch)
            end_event.record()
            torch.cuda.synchronize()
            elapsed_ms = start_event.elapsed_time(end_event)

            # Skip warmup batches
            if i // batch_size >= warmup_batches:
                latencies.append(elapsed_ms)

            # Mean pool over sequence length (dim=1)
            # Handle models that may return dicts, etc.
            if hasattr(outputs, "last_hidden_state"):
                hidden = outputs.last_hidden_state
            else:
                # Fallback for some models (e.g., if they only output logits)
                hidden = outputs[0]
            emb = hidden.mean(dim=1)   # [batch, hidden_dim]
            all_embeddings.append(emb.cpu().numpy())

    # Aggregate hardware stats
    peak_mem = torch.cuda.max_memory_allocated() / (1024**2)  # MiB
    avg_latency = np.mean(latencies) if latencies else 0.0

    return np.concatenate(all_embeddings, axis=0), peak_mem, avg_latency

train_emb, train_peak_mem, train_latency = get_embeddings(train_texts)
test_emb,  test_peak_mem,  test_latency  = get_embeddings(test_texts)

print(f"   Embedding shape: train {train_emb.shape}, test {test_emb.shape}")
print(f"   Peak GPU memory: {test_peak_mem:.1f} MiB, avg latency: {test_latency:.2f} ms")

# ---------------------------------------------------------------------------
# 4. Train probe & evaluate
# ---------------------------------------------------------------------------
print("[4/5] Training linear probe ...")
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

clf = LogisticRegression(
    max_iter=PROBE_MAX_ITER,
    random_state=RANDOM_SEED,
    class_weight="balanced",    # good for imbalanced genomics data
)
clf.fit(train_emb, train_labels)

# Predictions
test_pred = clf.predict(test_emb)
test_score = clf.decision_function(test_emb)  # For AUPRC

mcc = matthews_corrcoef(test_labels, test_pred)
auprc = average_precision_score(test_labels, test_score)

print(f"   MCC: {mcc:.4f}, AUPRC: {auprc:.4f}")

# ---------------------------------------------------------------------------
# 5. Report & save artifact
# ---------------------------------------------------------------------------
print("[5/5] Saving result ...")

result = {
    "model": MODEL_NAME,
    "task": TASK_NAME,
    "evaluation_protocol": "linear_probing",
    "max_sequence_length": MAX_LEN,
    "batch_size": BATCH_SIZE,
    "probe": {
        "type": "logistic_regression",
        "max_iter": PROBE_MAX_ITER,
        "class_weight": "balanced",
    },
    "metrics": {
        "mcc": round(mcc, 4),
        "auprc": round(auprc, 4),
    },
    "hardware": {
        "gpu": torch.cuda.get_device_name(0) if DEVICE == "cuda" else "CPU",
        "peak_memory_mb": round(test_peak_mem, 1),
        "avg_inference_latency_ms": round(test_latency, 2),
        "warmup_batches": 2,
        "measured_batches": len(test_texts) // BATCH_SIZE - 2,
    },
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "genobench_version": "0.1.0-dev",
    "env": {
        "python": os.sys.version,
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
    },
}

# Print a clean table row
print("\n" + "="*80)
print(f"{'Model':<40} {'Task':<20} {'MCC':>6} {'AUPRC':>6} {'GPU Mem (MiB)':>14} {'Lat (ms)':>8}")
print("-"*80)
print(f"{MODEL_NAME:<40} {TASK_NAME:<20} {mcc:6.4f} {auprc:6.4f} {test_peak_mem:14.1f} {test_latency:8.2f}")
print("="*80)

# Save JSON artifact
os.makedirs("results", exist_ok=True)
out_path = os.path.join("results", f"{TASK_NAME}_{MODEL_NAME.replace('/', '_')}.json")
with open(out_path, "w") as f:
    json.dump(result, f, indent=2)
print(f"\nResult saved to {out_path}")