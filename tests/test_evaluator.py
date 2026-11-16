import unittest
from unittest import mock

from genobench.evaluator import run_evaluation


class TestEvaluatorPipeline(unittest.TestCase):
    def test_dummy_end_to_end_run_records_protocol(self):
        """Runs the full pipeline on the dummy model/task without touching results/."""
        with mock.patch("genobench.evaluator.save_result_to_jsonl") as save:
            run_evaluation("dummy_gfm", "dummy_task", batch_size=2)

        save.assert_called_once()
        result = save.call_args[0][0]
        self.assertEqual(result["model"], "dummy_gfm")
        self.assertEqual(result["task"], "dummy_task")
        self.assertEqual(result["evaluation_protocol"], "linear_probing")
        # Embeddings are standardized before the probe so all models share one protocol
        self.assertTrue(result["feature_standardization"])
        self.assertIn("mcc", result["metrics"])
        self.assertIn("auprc", result["metrics"])
        self.assertIn("peak_memory_mib", result["hardware"])


if __name__ == "__main__":
    unittest.main()
