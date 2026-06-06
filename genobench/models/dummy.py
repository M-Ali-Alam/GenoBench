import random
from . import register_model

@register_model("dummy_gfm")
class DummyModel:
    def __init__(self):
        self.device = "cpu"
        print("[System] Initialized DummyModel.")

    def predict(self, sequence: str) -> float:
        """Mock prediction simulating a model forward pass."""
        # Returns a random confidence score between 0 and 1
        return random.uniform(0.0, 1.0)