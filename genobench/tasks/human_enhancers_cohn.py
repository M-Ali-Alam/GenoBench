from typing import List, Tuple
from genomic_benchmarks.dataset_getters.pytorch_datasets import GenomicClfDataset
from . import register_task

@register_task("human_enhancers_cohn")
class HumanEnhancersCohnTask:
    def __init__(self, **kwargs):
        self.train_dataset = None
        self.test_dataset = None

    def get_train_data(self) -> Tuple[List[str], List[int]]:
        if self.train_dataset is None:
            self.train_dataset = GenomicClfDataset("human_enhancers_cohn", "train")
        
        sequences = [self.train_dataset[i][0] for i in range(len(self.train_dataset))]
        labels = [self.train_dataset[i][1] for i in range(len(self.train_dataset))]
        return sequences, labels

    def get_test_data(self) -> Tuple[List[str], List[int]]:
        if self.test_dataset is None:
            self.test_dataset = GenomicClfDataset("human_enhancers_cohn", "test")
        
        sequences = [self.test_dataset[i][0] for i in range(len(self.test_dataset))]
        labels = [self.test_dataset[i][1] for i in range(len(self.test_dataset))]
        return sequences, labels
