import torch
import numpy as np
from typing import List, Optional
from transformers import AutoModelForMaskedLM, AutoTokenizer
from . import register_model
from .base import BaseGFM

@register_model("nucleotide_transformer")
class NucleotideTransformerGFM(BaseGFM):
    """
    Wrapper for Nucleotide Transformer v2 (50M Multi-Species).
    Loads InstaDeepAI/nucleotide-transformer-v2-50m-multi-species and extracts mean-pooled embeddings.
    """
    def __init__(self, **kwargs):
        self.checkpoint = kwargs.get("checkpoint", "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species")
        self.device = kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = kwargs.get("max_length", 1024)
        
        print(f"[Model] Initializing NucleotideTransformer (50M) on {self.device.upper()}...")
        print(f"        -> Checkpoint: {self.checkpoint}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.checkpoint, 
            trust_remote_code=True
        )
        
        # Use AutoModelForMaskedLM in float32 for ESM architecture compatibility
        self.model = AutoModelForMaskedLM.from_pretrained(
            self.checkpoint, 
            trust_remote_code=True,
            torch_dtype=torch.float32
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
                        outputs = self.model(**inputs, output_hidden_states=True)
            else:
                with torch.no_grad():
                    outputs = self.model(**inputs, output_hidden_states=True)
            
            if hasattr(outputs, "hidden_states") and outputs.hidden_states:
                hidden = outputs.hidden_states[-1]
            elif hasattr(outputs, "last_hidden_state"):
                hidden = outputs.last_hidden_state
            else:
                hidden = outputs[0]
                
            pooled = hidden.mean(dim=1)
            embeddings.append(pooled.cpu().to(torch.float32).numpy())

        return np.concatenate(embeddings, axis=0)
