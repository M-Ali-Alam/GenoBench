from typing import List, Optional
import numpy as np

class BaseGFM:
    def get_embeddings(self, texts: List[str], batch_size: int = 16, probe: Optional[any] = None) -> np.ndarray:
        """
        Extracts frozen representations (embeddings) for a list of DNA sequences.
        Returns a numpy array of shape (len(texts), hidden_dim) on CPU.
        
        Args:
            texts: List of genomic DNA sequences (strings).
            batch_size: Number of sequences to process in a batch.
            probe: Optional HardwareProbe context manager for timing individual forward passes.
        """
        raise NotImplementedError
