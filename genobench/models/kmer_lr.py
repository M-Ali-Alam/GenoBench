import itertools
from typing import List, Optional
import numpy as np
from . import register_model
from .base import BaseGFM

@register_model("kmer_lr")
class KMerBaselineGFM(BaseGFM):
    """
    Classical k-mer count frequency baseline model.
    Extracts normalized 3-mer (64-dimensional) frequency vectors for DNA sequences on CPU.
    """
    def __init__(self, k: int = 3, **kwargs):
        self.k = k
        bases = ['A', 'C', 'G', 'T']
        self.kmers = [''.join(p) for p in itertools.product(bases, repeat=self.k)]
        self.kmer_to_idx = {kmer: idx for idx, kmer in enumerate(self.kmers)}
        self.num_kmers = len(self.kmers)  # 4^3 = 64
        
        print(f"[Model] Initialized KMerBaselineGFM (k={self.k}, dim={self.num_kmers}).")

    def _extract_sequence_vector(self, seq: str) -> np.ndarray:
        seq = seq.upper()
        counts = np.zeros(self.num_kmers, dtype=np.float32)
        n = len(seq)
        if n < self.k:
            return counts
        
        valid_kmers = 0
        for i in range(n - self.k + 1):
            kmer = seq[i : i + self.k]
            if kmer in self.kmer_to_idx:
                counts[self.kmer_to_idx[kmer]] += 1.0
                valid_kmers += 1
                
        if valid_kmers > 0:
            counts /= valid_kmers
        return counts

    def get_embeddings(self, texts: List[str], batch_size: int = 16, probe: Optional[any] = None) -> np.ndarray:
        embeddings = []
        num_samples = len(texts)
        
        for i in range(0, num_samples, batch_size):
            batch_texts = texts[i : i + batch_size]
            
            if probe is not None:
                with probe.measure_batch():
                    batch_vecs = [self._extract_sequence_vector(t) for t in batch_texts]
            else:
                batch_vecs = [self._extract_sequence_vector(t) for t in batch_texts]
                
            embeddings.append(np.array(batch_vecs, dtype=np.float32))
            
        return np.concatenate(embeddings, axis=0)
