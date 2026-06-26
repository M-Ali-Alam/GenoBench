import random
from . import register_model

@register_model("dummy_gfm")
class DummyModel:
    def __init__(self, **kwargs):
        # Extract kwargs with safe fallbacks
        self.device = kwargs.get("device", "cpu")
        self.checkpoint = kwargs.get("checkpoint", "default-weights")
        self.batch_size = kwargs.get("batch_size", 1)
        
        print(f"[Model] Initialized DummyModel.")
        print(f"        -> Checkpoint: {self.checkpoint}")
        print(f"        -> Device: {self.device}")
        print(f"        -> Batch Size: {self.batch_size}")

    def predict(self, sequence: str) -> float:
        """Mock prediction simulating a model forward pass."""
        return random.uniform(0.0, 1.0)