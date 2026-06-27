# Extensibility Guide: Adding Models and Tasks

GenoBench is designed to be easily extensible. You can register new Genomic Foundation Models (GFMs) or binary classification tasks by following the guidelines below.

---

## 1. Adding a New Model

To register a new GFM:

### Step 1: Create the model wrapper
Create a new file in `genobench/models/` (e.g., `genobench/models/dnabert.py`).
The class must:
1.  Inherit from the `BaseGFM` interface in [base.py](../genobench/models/base.py).
2.  Use the `@register_model("model_name")` decorator.
3.  Implement `__init__` to load the tokenizer and model onto the target device.
4.  Implement `get_embeddings` to process batches of sequences, wrap the forward pass inside `probe.measure_batch()` (if a probe is passed), mean-pool the sequence representations, and return a NumPy array.

#### Example:
```python
import torch
import numpy as np
from typing import List, Optional
from transformers import AutoModel, AutoTokenizer
from . import register_model
from .base import BaseGFM

@register_model("my_new_gfm")
class MyNewGFM(BaseGFM):
    def __init__(self, **kwargs):
        self.checkpoint = kwargs.get("checkpoint", "path-to-huggingface-checkpoint")
        self.device = kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = kwargs.get("max_length", 1024)
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)
        self.model = AutoModel.from_pretrained(self.checkpoint).to(self.device)
        self.model.eval()

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = "[PAD]"

    def get_embeddings(self, texts: List[str], batch_size: int = 16, probe: Optional[any] = None) -> np.ndarray:
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            
            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length
            ).to(self.device)
            
            if probe is not None:
                with probe.measure_batch():
                    with torch.no_grad():
                        outputs = self.model(**inputs)
            else:
                with torch.no_grad():
                    outputs = self.model(**inputs)
            
            # Extract sequence representation
            hidden_states = outputs[0]
            pooled = hidden_states.mean(dim=1)
            embeddings.append(pooled.cpu().numpy())
            
        return np.concatenate(embeddings, axis=0)
```

### Step 2: Register in Init
Import your new module in [genobench/models/__init__.py](../genobench/models/__init__.py) to execute the decorator registration:
```python
from . import my_new_gfm
```

---

## 2. Adding a New Task

To register a new binary classification task:

### Step 1: Create the task loader
Create a new file in `genobench/tasks/` (e.g., `genobench/tasks/drosophila_enhancers.py`).
The class must:
1.  Use the `@register_task("task_name")` decorator.
2.  Implement `get_train_data(self) -> Tuple[List[str], List[int]]` returning training sequences and integer labels.
3.  Implement `get_test_data(self) -> Tuple[List[str], List[int]]` returning testing sequences and integer labels.

#### Example:
```python
from typing import List, Tuple
from genomic_benchmarks.dataset_getters.pytorch_datasets import GenomicClfDataset
from . import register_task

@register_task("drosophila_enhancers")
class DrosophilaEnhancersTask:
    def __init__(self, **kwargs):
        self.train_dataset = None
        self.test_dataset = None

    def get_train_data(self) -> Tuple[List[str], List[int]]:
        if self.train_dataset is None:
            self.train_dataset = GenomicClfDataset("drosophila_enhancers_stark", "train")
        sequences = [self.train_dataset[i][0] for i in range(len(self.train_dataset))]
        labels = [self.train_dataset[i][1] for i in range(len(self.train_dataset))]
        return sequences, labels

    def get_test_data(self) -> Tuple[List[str], List[int]]:
        if self.test_dataset is None:
            self.test_dataset = GenomicClfDataset("drosophila_enhancers_stark", "test")
        sequences = [self.test_dataset[i][0] for i in range(len(self.test_dataset))]
        labels = [self.test_dataset[i][1] for i in range(len(self.test_dataset))]
        return sequences, labels
```

### Step 2: Register in Init
Import your new task module in [genobench/tasks/__init__.py](../genobench/tasks/__init__.py) to execute the decorator registration:
```python
from . import drosophila_enhancers
```
