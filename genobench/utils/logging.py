import json
import os

def save_result_to_jsonl(result: dict, filepath: str = "results/evaluation_runs.jsonl") -> None:
    """
    Appends a benchmarking run result as a single JSON line to the specified file.
    Creates any parent directories if they don't exist.
    """
    if not filepath:
        return
    
    dir_path = os.path.dirname(filepath)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
        
    with open(filepath, "a") as f:
        f.write(json.dumps(result) + "\n")
