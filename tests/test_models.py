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

if __name__ == "__main__":
    unittest.main()
