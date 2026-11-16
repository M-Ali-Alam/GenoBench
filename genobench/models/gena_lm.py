import torch
import numpy as np
from typing import List, Optional
from transformers import AutoModel, AutoTokenizer
from . import register_model
from .base import BaseGFM

# Gena-LM's BPE tokenizer and BERT context window are capped at 512 tokens (~4 bp/token).
_MAX_TOKENS = 512


@register_model("gena_lm")
class GenaLMWrapper(BaseGFM):
    """
    Gena-LM (BERT-base, ~110M parameters, BPE tokenizer, T2T human genome pretraining).

    The checkpoint ships custom modeling code, so loading uses trust_remote_code=True and
    checkpoints are restricted to a whitelist. AutoModel resolves to BertForMaskedLM, which
    only returns logits, so hidden states are requested explicitly and the last layer is
    mean-pooled over non-padding tokens.
    """

    def __init__(self, **kwargs):
        self.checkpoint = kwargs.get("checkpoint", "AIRI-Institute/gena-lm-bert-base-t2t")

        # Security check: whitelist checkpoints to prevent RCE via untrusted remote code.
        # The -multi variant is excluded because it only ships pickled pytorch_model.bin.
        allowed_checkpoints = {"AIRI-Institute/gena-lm-bert-base-t2t"}
        if self.checkpoint not in allowed_checkpoints:
            raise ValueError(
                f"Untrusted checkpoint '{self.checkpoint}'. "
                f"Allowed checkpoints for Gena-LM: {allowed_checkpoints}"
            )

        self.device = kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        # max_length is in tokens for the tokenizer; cap it at the model's position limit.
        self.max_length = min(int(kwargs.get("max_length", 1024)), _MAX_TOKENS)

        print(f"[Model] Initializing Gena-LM on {self.device.upper()}...")
        print(f"        -> Checkpoint: {self.checkpoint}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            self.checkpoint,
            trust_remote_code=True,
            torch_dtype=torch.float16 if "cuda" in self.device else torch.float32,
        ).to(self.device)
        self.model.eval()

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = "[PAD]"

    def get_embeddings(self, texts: List[str], batch_size: int = 16, probe: Optional[any] = None) -> np.ndarray:
        embeddings = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]

            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            )
            # The BERT forward is only exercised with input_ids/attention_mask.
            inputs.pop("token_type_ids", None)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            if probe is not None:
                with probe.measure_batch():
                    with torch.no_grad():
                        outputs = self.model(**inputs, output_hidden_states=True)
            else:
                with torch.no_grad():
                    outputs = self.model(**inputs, output_hidden_states=True)

            # Pool in float32: Gena-LM is pre-LayerNorm with no final norm, so last-layer
            # activations reach ~1e4 and summing them over tokens overflows fp16 (max 65504).
            hidden = outputs.hidden_states[-1].to(torch.float32)
            mask = inputs["attention_mask"].unsqueeze(-1).to(torch.float32)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            embeddings.append(pooled.cpu().numpy())

        return np.concatenate(embeddings, axis=0)
