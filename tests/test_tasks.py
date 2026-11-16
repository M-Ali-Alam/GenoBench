import unittest
from genobench.tasks import TASK_REGISTRY, get_task

class TestTaskRegistry(unittest.TestCase):
    def test_all_tasks_registered(self):
        expected_tasks = [
            "dummy_task",
            "human_vs_worm",
            "human_enhancers_cohn",
            "coding_vs_intergenic",
            "human_nontarget_promoters",
            "human_ocr_ensembl",
            "drosophila_enhancers_stark",
            "mouse_enhancers_ensembl",
            "dummy_mouse_enhancers_ensembl",  # deprecated alias
            "human_enhancers_ensembl",
        ]
        for task_name in expected_tasks:
            self.assertIn(task_name, TASK_REGISTRY, f"Task '{task_name}' should be registered in TASK_REGISTRY")

    def test_real_tasks_expose_train_and_test_loaders(self):
        # The evaluator needs get_train_data/get_test_data; checked without downloading data.
        for name, cls in TASK_REGISTRY.items():
            if name == "dummy_task":
                continue  # legacy single-split stub (get_data)
            self.assertTrue(callable(getattr(cls, "get_train_data", None)), f"{name} lacks get_train_data")
            self.assertTrue(callable(getattr(cls, "get_test_data", None)), f"{name} lacks get_test_data")

    def test_mouse_enhancer_alias_resolves_to_same_task(self):
        self.assertIs(TASK_REGISTRY["mouse_enhancers_ensembl"], TASK_REGISTRY["dummy_mouse_enhancers_ensembl"])

    def test_invalid_task_raises_error(self):
        with self.assertRaises(ValueError) as context:
            get_task("non_existent_task")
        self.assertIn("non_existent_task", str(context.exception))

if __name__ == "__main__":
    unittest.main()
