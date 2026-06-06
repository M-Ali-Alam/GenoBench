import argparse
import sys
from genobench.evaluator import run_evaluation

def main():
    parser = argparse.ArgumentParser(description="GenoBench: GFM Evaluation Framework")
    parser.add_argument("--model", type=str, required=True, help="Registered model name (e.g., dummy_gfm)")
    parser.add_argument("--task", type=str, required=True, help="Task name (e.g., dummy_task)")
    
    args = parser.parse_args()
    
    try:
        run_evaluation(args.model, args.task)
    except Exception as e:
        print(f"[Error] Pipeline failed: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()