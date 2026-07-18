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
            "dummy_mouse_enhancers_ensembl",
            "human_enhancers_ensembl",
        ]
        for task_name in expected_tasks:
            self.assertIn(task_name, TASK_REGISTRY, f"Task '{task_name}' should be registered in TASK_REGISTRY")

    def test_invalid_task_raises_error(self):
        with self.assertRaises(ValueError) as context:
            get_task("non_existent_task")
        self.assertIn("non_existent_task", str(context.exception))

if __name__ == "__main__":
    unittest.main()
