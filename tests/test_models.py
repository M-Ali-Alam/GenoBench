import unittest
import numpy as np
from genobench.models import MODEL_REGISTRY, get_model

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

if __name__ == "__main__":
    unittest.main()
