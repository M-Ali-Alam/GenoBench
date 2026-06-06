import json
from genobench.models import get_model
from genobench.tasks import get_task
from genobench.metrics.classification import ClassificationEvaluator

def run_evaluation(model_name: str, task_name: str, **kwargs) -> None:
    print(f"--- Starting Evaluation: {model_name} on {task_name} ---")
    
    # 1. Initialize core system modules dynamically
    model = get_model(model_name, **kwargs)
    task = get_task(task_name, **kwargs)
    metric_tracker = ClassificationEvaluator()
    
    # 2. Data extraction
    dataset = task.get_data()

    # 3. Stream processing
    print("[Pipeline] Processing sequences...")
    for sequence, true_label in dataset:
        prediction = model.predict(sequence)
        metric_tracker.update(true_label, prediction)
        
    # 4. Final analytical execution
    metrics = metric_tracker.compute_metrics()
    
    print("\n--- Evaluation Metrics Summary ---")
    print(json.dumps(metrics, indent=4))
    print("-----------------------------------\n")