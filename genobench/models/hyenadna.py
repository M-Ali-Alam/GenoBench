import torch
import numpy as np
from typing import List, Optional
from transformers import AutoModel, AutoTokenizer
from . import register_model
from .base import BaseGFM

@register_model("hyenadna")
class HyenaDNAWrapper(BaseGFM):
    def __init__(self, **kwargs):
        # 1. Extract configurations with safe fallbacks
        self.checkpoint = kwargs.get("checkpoint", "LongSafari/hyenadna-tiny-1k-seqlen-hf")
        
        # Security check: Whitelist allowed checkpoints to prevent RCE via untrusted remote code
        allowed_checkpoints = {
            "LongSafari/hyenadna-tiny-1k-seqlen-hf",
            "LongSafari/hyenadna-small-32k-seqlen-hf",
            "LongSafari/hyenadna-medium-160k-seqlen-hf",
            "LongSafari/hyenadna-large-1m-seqlen-hf",
        }
        if self.checkpoint not in allowed_checkpoints:
            raise ValueError(
                f"Untrusted checkpoint '{self.checkpoint}'. "
                f"Allowed checkpoints for HyenaDNA: {allowed_checkpoints}"
            )
            
        self.device = kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = kwargs.get("max_length", 1024)
        
        print(f"[Model] Initializing HyenaDNA on {self.device.upper()}...")
        print(f"        -> Checkpoint: {self.checkpoint}")

        # 2. Load Tokenizer and Base Model
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.checkpoint, 
            trust_remote_code=True
        )
        
        # Load base model for representation extraction, optionally in float16 for VRAM savings
        self.model = AutoModel.from_pretrained(
            self.checkpoint, 
            trust_remote_code=True,
            torch_dtype=torch.float16 if "cuda" in self.device else torch.float32
        ).to(self.device)
        
        self.model.eval()

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token if self.tokenizer.eos_token else "[PAD]"

    def get_embeddings(self, texts: List[str], batch_size: int = 16, probe: Optional[any] = None) -> np.ndarray:
        embeddings = []
        num_samples = len(texts)

        for i in range(0, num_samples, batch_size):
            batch_texts = texts[i : i + batch_size]
            
            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            if probe is not None:
                with probe.measure_batch():
                    with torch.no_grad():
                        outputs = self.model(**inputs)
            else:
                with torch.no_grad():
                    outputs = self.model(**inputs)
            
            if hasattr(outputs, "last_hidden_state"):
                hidden = outputs.last_hidden_state
            else:
                hidden = outputs[0]
                
            pooled = hidden.mean(dim=1)
            # Ensure float32 representation for scikit-learn probing compatibility
            embeddings.append(pooled.cpu().to(torch.float32).numpy())

        return np.concatenate(embeddings, axis=0)