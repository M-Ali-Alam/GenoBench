from typing import List, Optional
import numpy as np
from . import register_model
from .base import BaseGFM

@register_model("dummy_gfm")
class DummyModel(BaseGFM):
    def __init__(self, **kwargs):
        self.device = kwargs.get("device", "cpu")
        self.checkpoint = kwargs.get("checkpoint", "default-weights")
        self.batch_size = kwargs.get("batch_size", 1)
        self.embedding_dim = kwargs.get("embedding_dim", 64)
        
        print(f"[Model] Initialized DummyModel.")
        print(f"        -> Checkpoint: {self.checkpoint}")
        print(f"        -> Device: {self.device}")

    def get_embeddings(self, texts: List[str], batch_size: int = 16, probe: Optional[any] = None) -> np.ndarray:
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            if probe is not None:
                with probe.measure_batch():
                    # Generate deterministic dummy embeddings based on sequence length/content seed
                    rng = np.random.default_rng(len(batch_texts))
                    batch_emb = rng.standard_normal((len(batch_texts), self.embedding_dim)).astype(np.float32)
            else:
                rng = np.random.default_rng(len(batch_texts))
                batch_emb = rng.standard_normal((len(batch_texts), self.embedding_dim)).astype(np.float32)
            embeddings.append(batch_emb)
        return np.concatenate(embeddings, axis=0)