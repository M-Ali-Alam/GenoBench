# Joint Hardware Profiling Methodology

GenoBench measures biological capability in conjunction with physical hardware consumption. Measuring hardware accurately in deep learning frameworks requires addressing bottlenecks like asynchronous GPU execution and system caching. 

This document describes the design and methodology behind GenoBench's hardware profiling system.

---

## The Asynchronous Execution Problem

PyTorch executes CUDA operations asynchronously. When a line of code like `outputs = model(**inputs)` is executed, the CPU submits the operation to the GPU queue and immediately proceeds to the next line of code. 

If you measure latency using standard CPU timers (e.g., `time.perf_counter()`):
*   You will only measure the time it took the CPU to *queue* the operations, not the time it took the GPU to *execute* them.
*   If you force synchronization using `torch.cuda.synchronize()`, you introduce a massive performance overhead that slows down execution and pollutes the metrics.

---

## The GenoBench Solution: `HardwareProbe`

GenoBench implements a context-driven tracking module in [probe.py](../genobench/hardware/probe.py) called `HardwareProbe`. It measures memory and latency using the following protocols:

### 1. CUDA Event-Based Timing
Instead of wall-clock timers, `HardwareProbe` uses CUDA Events. These are markers recorded directly in the GPU stream:
```python
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

start.record()
yield # Execute forward pass
end.record()

torch.cuda.synchronize()
elapsed_ms = start.elapsed_time(end)
```
This records precise GPU execution intervals. By wrapping each batch inside the model's extraction loop, the probe accumulates individual batch latencies.

### 2. Peak GPU Memory Tracking
Peak GPU memory allocation (VRAM) is captured over the lifetime of the `HardwareProbe` context:
*   At context entry (`__enter__`), GPU caches are cleared (`torch.cuda.empty_cache()`) and memory statistics are reset (`torch.cuda.reset_peak_memory_stats()`).
*   During extraction, PyTorch's internal allocator tracks the maximum memory requested.
*   At context exit (`__exit__`), the peak memory is read via `torch.cuda.max_memory_allocated() / (1024 ** 2)` and reported in MiB.

---

## Benchmark Run Pipeline

To ensure the hardware metrics are accurate and not skewed by startup costs, the following steps are performed:
1.  **Warmup Pass**: Before measurements begin, the model is run with a small batch of dummy sequences. This initializes the tokenizer, triggers lazy loading of PyTorch/CUDA libraries, compiles custom Triton attention kernels, and pre-allocates standard model weights in GPU VRAM.
2.  **Continuous Profiling**: The test sequence embeddings are extracted. Since the model processes multiple batches sequentially, the probe tracks every single batch forward pass, computing:
    *   **Average Batch Inference Latency**: The mean of all timed batch forward passes.
    *   **Peak VRAM**: The highest active VRAM allocation measured during the entire test set extraction run.
