import json
import os
import time
import torch
import numpy as np
from datetime import datetime, timezone
from sklearn.linear_model import LogisticRegression

from genobench.models import get_model
from genobench.tasks import get_task
from genobench.metrics.classification import ClassificationEvaluator
from genobench.hardware.probe import HardwareProbe
from genobench.utils.logging import save_result_to_jsonl

def run_evaluation(model_name: str, task_name: str, **kwargs) -> None:
    print(f"\n--- Starting GenoBench V1 Evaluation: {model_name} on {task_name} ---")
    
    # 1. Retrieve CLI configuration parameters
    batch_size = int(kwargs.get("batch_size", 16))
    max_length = int(kwargs.get("max_length", 1024))
    max_train_samples = kwargs.get("max_train_samples")
    max_test_samples = kwargs.get("max_test_samples")
    if max_train_samples is not None:
        max_train_samples = int(max_train_samples)
    if max_test_samples is not None:
        max_test_samples = int(max_test_samples)

    # 2. Initialize Core Modules Dynamically
    task = get_task(task_name, **kwargs)
    metric_tracker = ClassificationEvaluator()

    # 3. Retrieve, shuffle, and slice task dataset splits
    if hasattr(task, "get_train_data") and hasattr(task, "get_test_data"):
        print("[Pipeline] Fetching train and test splits...")
        train_texts, train_labels = task.get_train_data()
        test_texts, test_labels = task.get_test_data()
        
        # Shuffle using a fixed seed for reproducibility
        rng = np.random.default_rng(42)
        train_indices = rng.permutation(len(train_texts))
        train_texts = [train_texts[idx] for idx in train_indices]
        train_labels = [train_labels[idx] for idx in train_indices]
        
        test_indices = rng.permutation(len(test_texts))
        test_texts = [test_texts[idx] for idx in test_indices]
        test_labels = [test_labels[idx] for idx in test_indices]
    else:
        # Fallback compatibility with old mock/dummy tasks
        print("[Pipeline] Fetching single split (legacy fallback)...")
        data = task.get_data()
        train_texts = [d[0] for d in data]
        train_labels = [d[1] for d in data]
        test_texts = train_texts
        test_labels = train_labels

    if max_train_samples is not None:
        print(f"           -> Slicing train dataset to {max_train_samples} samples")
        train_texts = train_texts[:max_train_samples]
        train_labels = train_labels[:max_train_samples]
        
    if max_test_samples is not None:
        print(f"           -> Slicing test dataset to {max_test_samples} samples")
        test_texts = test_texts[:max_test_samples]
        test_labels = test_labels[:max_test_samples]

    print(f"           -> Total train samples: {len(train_texts)}")
    print(f"           -> Total test samples: {len(test_texts)}")

    # 4. Instantiate Model
    if "model_kwargs" in kwargs:
        mk = kwargs["model_kwargs"]
        if isinstance(mk, str):
            try:
                import json
                mk = json.loads(mk)
            except Exception:
                pass
        if isinstance(mk, dict) and "checkpoint" in mk:
            import warnings
            warnings.warn(
                "Ignoring 'checkpoint' key in model_kwargs. Please use the --checkpoint argument directly.",
                UserWarning
            )

    kwargs["max_length"] = max_length
    model = get_model(model_name, **kwargs)

    # 5. Warm up the GFM (compile custom architectures, cache tokenizer, warm CUDA)
    print("[Pipeline] Warming up model...")
    warmup_texts = ["ATCG" * 100] * min(batch_size, 2)
    model.get_embeddings(warmup_texts, batch_size=len(warmup_texts))

    # 6. Extract frozen embeddings for train split
    print(f"[Pipeline] Extracting training representations...")
    train_embeddings = model.get_embeddings(train_texts, batch_size=batch_size)

    # 7. Extract frozen embeddings for test split with hardware metrics
    print(f"[Pipeline] Extracting testing representations (with hardware probe)...")
    with HardwareProbe() as probe:
        test_embeddings = model.get_embeddings(test_texts, batch_size=batch_size, probe=probe)

    # 8. Train the linear probe classifier
    print(f"[Pipeline] Training linear probe (LogisticRegression)...")
    clf = LogisticRegression(
        class_weight="balanced",
        random_state=42,
        max_iter=1000
    )
    clf.fit(train_embeddings, train_labels)

    # 9. Evaluate the linear probe
    print(f"[Pipeline] Running inference on test representations...")
    test_predictions = clf.predict(test_embeddings)
    if len(clf.classes_) > 1:
        test_probabilities = clf.predict_proba(test_embeddings)[:, 1]
    else:
        test_probabilities = np.zeros(len(test_embeddings))

    for true_label, prob in zip(test_labels, test_probabilities):
        metric_tracker.update(true_label, prob)

    metrics = metric_tracker.compute_metrics()

    # 10. Compile and serialize results
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    
    result = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": model_name,
        "task": task_name,
        "evaluation_protocol": "linear_probing",
        "hyperparameters": {
            "max_sequence_length": max_length,
            "batch_size": batch_size,
            "max_train_samples": max_train_samples,
            "max_test_samples": max_test_samples,
        },
        "metrics": metrics,
        "hardware": {
            "gpu": gpu_name,
            "peak_memory_mib": round(probe.peak_memory, 2),
            "avg_batch_inference_latency_ms": round(probe.mean_latency, 2),
            "total_batches_measured": len(probe.batch_latencies),
        },
        "env": {
            "python": os.sys.version.split()[0],
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
        }
    }

    save_result_to_jsonl(result)

    print("\n--- Evaluation Metrics Summary ---")
    print(json.dumps(metrics, indent=4))
    print("--- Hardware Usage Summary ---")
    print(f"Peak GPU Memory: {probe.peak_memory:.2f} MiB")
    print(f"Average Batch Inference Latency: {probe.mean_latency:.2f} ms")
    print("-----------------------------------\n")