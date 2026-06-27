import json
import os

def save_result_to_jsonl(result: dict, filepath: str = "results/evaluation_runs.jsonl") -> None:
    """
    Appends a benchmarking run result as a single JSON line to the specified file.
    Always saves inside the project's root results/ folder unless an absolute path is specified.
    """
    if not filepath:
        return
        
    # If the path is relative, resolve it relative to the project root directory
    # (two levels up from this file's location: genobench/utils/logging.py)
    if not os.path.isabs(filepath):
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        filepath = os.path.join(project_root, filepath)
        
    dir_path = os.path.dirname(filepath)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
        
    with open(filepath, "a") as f:
        f.write(json.dumps(result) + "\n")
