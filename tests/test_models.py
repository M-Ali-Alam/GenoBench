import unittest
from types import SimpleNamespace
import numpy as np
import torch
from genobench.models import MODEL_REGISTRY, get_model
from genobench.models.gena_lm import GenaLMWrapper

class TestModelRegistry(unittest.TestCase):
    def test_all_models_registered(self):
        expected_models = [
            "dummy_gfm",
            "hyenadna",
            "hyenadna_small",
            "dnabert2",
            "kmer_lr",
            "nucleotide_transformer",
            "onehot_cnn",
            "gena_lm",
        ]
        for model_name in expected_models:
            self.assertIn(model_name, MODEL_REGISTRY, f"Model '{model_name}' should be registered in MODEL_REGISTRY")

    def test_kmer_lr_embedding_extraction(self):
        model = get_model("kmer_lr", k=3)
        sequences = ["ATGCGTACGTAGCTA", "CCGGAATTCCGGAAA"]
        embeddings = model.get_embeddings(sequences, batch_size=2)
        
        self.setIsInstance = True
        self.assertTrue(isinstance(embeddings, np.ndarray))
        self.assertEqual(embeddings.shape, (2, 64))

    def test_onehot_cnn_embeddings_shape_and_determinism(self):
        seqs = ["ATGCGTACGTAGCTA", "CCGGAATTCCGGAAANNNN", "A"]
        a = get_model("onehot_cnn", device="cpu", num_filters=32).get_embeddings(seqs, batch_size=2)
        b = get_model("onehot_cnn", device="cpu", num_filters=32).get_embeddings(seqs, batch_size=3)
        self.assertEqual(a.shape, (3, 64))
        self.assertTrue(np.isfinite(a).all())
        # Same seed -> same features, independent of batch size / padding
        np.testing.assert_allclose(a, b, atol=1e-5)

    def test_gena_lm_rejects_untrusted_checkpoint(self):
        with self.assertRaises(ValueError):
            get_model("gena_lm", checkpoint="someone/else-model")

    def test_gena_lm_pooling_ignores_padding_and_avoids_fp16_overflow(self):
        # Bypass __init__ (no download); feed a mocked tokenizer and model.
        model = object.__new__(GenaLMWrapper)
        model.device = "cpu"
        model.max_length = 512

        seq_len = 400
        mask = torch.zeros(2, seq_len, dtype=torch.long)
        mask[0, :] = 1          # sequence 0: all 400 tokens valid
        mask[1, :10] = 1        # sequence 1: 10 valid tokens, rest is padding
        model.tokenizer = lambda texts, **kw: {
            "input_ids": torch.zeros(2, seq_len, dtype=torch.long),
            "token_type_ids": torch.zeros(2, seq_len, dtype=torch.long),
            "attention_mask": mask,
        }

        # fp16 activations of ~1.5e4 (as in Gena-LM): summing 400 of them overflows fp16 (max 65504)
        hidden = torch.full((2, seq_len, 4), 15000.0, dtype=torch.float16)
        hidden[1, 10:, :] = 60000.0  # padding positions must not affect the mean
        model.model = lambda **kw: SimpleNamespace(hidden_states=[hidden])

        out = model.get_embeddings(["ACGT", "AC"], batch_size=2)
        self.assertEqual(out.shape, (2, 4))
        self.assertTrue(np.isfinite(out).all())
        np.testing.assert_allclose(out, 15000.0, rtol=1e-6)

if __name__ == "__main__":
    unittest.main()
