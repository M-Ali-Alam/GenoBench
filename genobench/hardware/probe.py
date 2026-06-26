import time
import torch
from contextlib import contextmanager
import numpy as np

class HardwareProbe:
    """
    Context manager to track peak GPU memory usage and individual batch inference latencies.
    """
    def __init__(self):
        self.batch_latencies = []
        self.peak_memory = 0.0

    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if torch.cuda.is_available():
            # Get peak memory allocated in MiB
            self.peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)
        else:
            self.peak_memory = 0.0

    @contextmanager
    def measure_batch(self):
        """
        Context manager to measure the latency of a single batch forward pass.
        Accumulates latencies in milliseconds.
        """
        if torch.cuda.is_available():
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            yield
            end.record()
            torch.cuda.synchronize()
            self.batch_latencies.append(start.elapsed_time(end))
        else:
            t0 = time.perf_counter()
            yield
            self.batch_latencies.append((time.perf_counter() - t0) * 1000.0)

    @property
    def mean_latency(self) -> float:
        """Returns the average batch latency in milliseconds."""
        return float(np.mean(self.batch_latencies)) if self.batch_latencies else 0.0
