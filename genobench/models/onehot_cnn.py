from typing import List, Optional
import numpy as np
import torch
import torch.nn as nn
from . import register_model
from .base import BaseGFM

_BASE_TO_IDX = {"A": 0, "C": 1, "G": 2, "T": 3}


@register_model("onehot_cnn")
class OneHotCNNBaseline(BaseGFM):
    """
    Frozen, randomly initialised one-hot CNN baseline.

    Sequences are one-hot encoded (A/C/G/T; any other base such as N is an all-zero
    column), passed through a fixed-seed random 1D conv layer + ReLU, and pooled over
    valid positions with global max and mean (2 * num_filters dims). Weights are never
    trained, so this fits the linear-probing protocol: it measures how much a generic
    convolutional architecture with no pretraining gives a linear probe, a lower bound
    for the pretrained GFMs. It is NOT an end-to-end trained CNN; that needs the
    fine-tuning mode planned for v1.1.
    """

    def __init__(self, **kwargs):
        self.num_filters = int(kwargs.get("num_filters", 256))
        self.kernel_size = int(kwargs.get("kernel_size", 9))
        self.max_length = int(kwargs.get("max_length", 1024))
        self.seed = int(kwargs.get("seed", 42))
        self.device = kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu")

        # Seed a local generator so global RNG state is untouched and runs are reproducible.
        gen = torch.Generator().manual_seed(self.seed)
        self.conv = nn.Conv1d(4, self.num_filters, self.kernel_size, padding=self.kernel_size // 2)
        with torch.no_grad():
            for p in self.conv.parameters():
                p.copy_(torch.randn(p.shape, generator=gen) * 0.1)
        self.conv = self.conv.to(self.device).eval()

        print(f"[Model] Initialized OneHotCNNBaseline on {self.device.upper()} "
              f"(filters={self.num_filters}, kernel={self.kernel_size}, frozen random init, seed={self.seed}).")

    def _encode(self, batch_texts: List[str]):
        seqs = [t.upper()[: self.max_length] for t in batch_texts]
        width = max(1, max(len(s) for s in seqs))
        x = np.zeros((len(seqs), 4, width), dtype=np.float32)
        mask = np.zeros((len(seqs), width), dtype=bool)
        for i, s in enumerate(seqs):
            mask[i, : len(s)] = True
            for j, ch in enumerate(s):
                idx = _BASE_TO_IDX.get(ch)
                if idx is not None:
                    x[i, idx, j] = 1.0
        return (torch.from_numpy(x).to(self.device), torch.from_numpy(mask).to(self.device))

    def get_embeddings(self, texts: List[str], batch_size: int = 16, probe: Optional[any] = None) -> np.ndarray:
        embeddings = []
        for i in range(0, len(texts), batch_size):
            x, mask = self._encode(texts[i : i + batch_size])

            def forward():
                with torch.no_grad():
                    h = torch.relu(self.conv(x))                       # (B, F, L)
                    m = mask.unsqueeze(1)                              # (B, 1, L)
                    h_max = h.masked_fill(~m, 0.0).amax(dim=2)         # ReLU output >= 0, so 0 is safe
                    h_mean = (h * m).sum(dim=2) / m.sum(dim=2).clamp(min=1)
                    return torch.cat([h_max, h_mean], dim=1)

            if probe is not None:
                with probe.measure_batch():
                    pooled = forward()
            else:
                pooled = forward()
            embeddings.append(pooled.cpu().to(torch.float32).numpy())
        return np.concatenate(embeddings, axis=0)
