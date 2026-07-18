from typing import List, Tuple
import numpy as np
from genomic_benchmarks.dataset_getters.pytorch_datasets import GenomicClfDataset
from . import register_task

@register_task("drosophila_enhancers_stark")
class DrosophilaEnhancersTask:
    def __init__(self, **kwargs):
        self.train_dataset = None
        self.test_dataset = None

    def get_train_data(self) -> Tuple[List[str], List[int]]:
        if self.train_dataset is None:
            self.train_dataset = GenomicClfDataset("drosophila_enhancers_stark", "train")
        
        sequences = [self.train_dataset[i][0] for i in range(len(self.train_dataset))]
        labels = [self.train_dataset[i][1] for i in range(len(self.train_dataset))]
        
        rng = np.random.default_rng(42)
        indices = rng.permutation(len(sequences))
        return [sequences[i] for i in indices], [labels[i] for i in indices]

    def get_test_data(self) -> Tuple[List[str], List[int]]:
        if self.test_dataset is None:
            self.test_dataset = GenomicClfDataset("drosophila_enhancers_stark", "test")
        
        sequences = [self.train_dataset[i][0] for i in range(len(self.train_dataset))] if self.test_dataset is None else [self.test_dataset[i][0] for i in range(len(self.test_dataset))]
        labels = [self.test_dataset[i][1] for i in range(len(self.test_dataset))]
        
        rng = np.random.default_rng(42)
        indices = rng.permutation(len(sequences))
        return [sequences[i] for i in indices], [labels[i] for i in indices]
