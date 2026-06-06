from typing import List, Tuple
from . import register_task

@register_task("dummy_task")
class DummyTask:
    def __init__(self, **kwargs):
        print("[System] Initialized DummyTask.")
        
    def get_data(self) -> List[Tuple[str, int]]:
        """Returns mock genomic sequences and their binary labels."""
        return [
            ("ATGCGTACGTAGCTA", 1),
            ("CCGGAATTCCGGAAA", 0),
            ("TTTAAAGCGCGCTTA", 1)
        ]