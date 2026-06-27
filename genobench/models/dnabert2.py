import sys
import torch
import numpy as np
from typing import List, Optional
from transformers import AutoModel, AutoTokenizer, AutoConfig
from . import register_model
from .base import BaseGFM

@register_model("dnabert2")
class DNABERT2Wrapper(BaseGFM):
    def __init__(self, **kwargs):
        self.checkpoint = kwargs.get("checkpoint", "zhihan1996/DNABERT-2-117M")
        
        # Security check: Whitelist allowed checkpoints to prevent RCE via untrusted remote code
        allowed_checkpoints = {
            "zhihan1996/DNABERT-2-117M"
        }
        if self.checkpoint not in allowed_checkpoints:
            raise ValueError(
                f"Untrusted checkpoint '{self.checkpoint}'. "
                f"Allowed checkpoints for DNABERT-2: {allowed_checkpoints}"
            )
            
        self.device = kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = kwargs.get("max_length", 1024)
        
        print(f"[Model] Initializing DNABERT-2 on {self.device.upper()}...")
        print(f"        -> Checkpoint: {self.checkpoint}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.checkpoint, 
            trust_remote_code=True
        )
        
        # Load and patch the config to prevent HF version 5 compatibility issues
        config = AutoConfig.from_pretrained(self.checkpoint, trust_remote_code=True)
        if not hasattr(config, "pad_token_id") or config.pad_token_id is None:
            config.pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0

        self.model = AutoModel.from_pretrained(
            self.checkpoint, 
            config=config,
            trust_remote_code=True
        ).to(self.device)
        self.model.eval()

        # Monkey-patch any imported bert_layers module to disable Flash Attention/Triton
        # and force standard PyTorch attention (due to Triton tl.dot trans_b compatibility bug)
        for name, module in list(sys.modules.items()):
            if name.endswith("bert_layers"):
                print(f"[DNABERT-2 Patch] Disabling Triton flash attention in: {name}")
                module.flash_attn_qkvpacked_func = None

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
            
            hidden_states = outputs[0]
            pooled = hidden_states.mean(dim=1)
            embeddings.append(pooled.cpu().numpy())

        return np.concatenate(embeddings, axis=0)
