import unittest
import numpy as np
from genobench.metrics.classification import ClassificationEvaluator
from genobench.models import get_model

class TestClassificationMetrics(unittest.TestCase):
    def test_empty_evaluator_raises_value_error(self):
        evaluator = ClassificationEvaluator()
        with self.assertRaises(ValueError):
            evaluator.compute_metrics()

    def test_perfect_predictions(self):
        evaluator = ClassificationEvaluator()
        evaluator.update(0, 0.1)
        evaluator.update(0, 0.2)
        evaluator.update(1, 0.8)
        evaluator.update(1, 0.9)
        
        metrics = evaluator.compute_metrics()
        
        self.assertEqual(metrics["sample_count"], 4)
        self.assertAlmostEqual(metrics["mcc"], 1.0)
        self.assertAlmostEqual(metrics["auprc"], 1.0)

    def test_imperfect_predictions(self):
        # We manually compute expected values for a specific case:
        # y_true = [0, 0, 1, 1]
        # y_pred_prob = [0.1, 0.6, 0.4, 0.9]
        # At threshold 0.5:
        # y_pred_binary = [0, 1, 0, 1]
        # True Negatives = 1, False Positives = 1
        # False Negatives = 1, True Positives = 1
        # MCC = (TP*TN - FP*FN) / sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
        # MCC = (1*1 - 1*1) / sqrt(2*2*2*2) = 0.0
        evaluator = ClassificationEvaluator()
        evaluator.update(0, 0.1)
        evaluator.update(0, 0.6)
        evaluator.update(1, 0.4)
        evaluator.update(1, 0.9)
        
        metrics = evaluator.compute_metrics()
        self.assertAlmostEqual(metrics["mcc"], 0.0)
        # Precision-recall curve for:
        # y_true = [0, 0, 1, 1]
        # y_pred = [0.1, 0.6, 0.4, 0.9]
        # At threshold 0.9: P=1.0, R=0.5
        # At threshold 0.6: P=0.5, R=0.5
        # At threshold 0.4: P=2/3, R=1.0
        # At threshold 0.1: P=2/4, R=1.0
        # Check that auprc is a float between 0.0 and 1.0
        self.assertTrue(0.0 <= metrics["auprc"] <= 1.0)

class TestModelRegistry(unittest.TestCase):
    def test_invalid_model_name_raises_error(self):
        with self.assertRaises(ValueError) as context:
            get_model("non_existent_gfm")
        self.assertIn("non_existent_gfm", str(context.exception))

if __name__ == "__main__":
    unittest.main()
