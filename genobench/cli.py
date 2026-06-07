import argparse
import sys
from typing import List, Dict, Any
from genobench.evaluator import run_evaluation

def parse_dynamic_kwargs(unknown_args: List[str]) -> Dict[str, Any]:
    """
    Converts unknown CLI args into a kwargs dictionary.
    Example: ['--checkpoint', 'tiny', '--batch-size', '32'] -> {'checkpoint': 'tiny', 'batch_size': 32}
    """
    kwargs = {}
    i = 0
    while i < len(unknown_args):
        arg = unknown_args[i]
        if arg.startswith("--"):
            # Clean the key: strip dashes and convert internal dashes to underscores
            key = arg.lstrip("-").replace("-", "_")
            
            # If there's a next argument and it's not another flag, it's the value
            if i + 1 < len(unknown_args) and not unknown_args[i+1].startswith("--"):
                val = unknown_args[i+1]
                # Basic type inference: cast to int or float if applicable
                if val.isdigit(): 
                    val = int(val)
                elif val.replace('.', '', 1).isdigit() and val.count('.') == 1: 
                    val = float(val)
                    
                kwargs[key] = val
                i += 2
            else:
                # If no value follows, treat it as a boolean toggle
                kwargs[key] = True
                i += 1
        else:
            i += 1
    return kwargs

def main():
    parser = argparse.ArgumentParser(description="GenoBench: GFM Evaluation Framework")
    # Our strict, required pipeline definitions
    parser.add_argument("--model", type=str, required=True, help="Registered model name")
    parser.add_argument("--task", type=str, required=True, help="Task name")
    
    # CRITICAL FIX: Use parse_known_args() instead of parse_args()
    args, unknown = parser.parse_known_args()
    
    # Convert the extra flags into a dictionary
    kwargs = parse_dynamic_kwargs(unknown)
    
    try:
        # Pass the dynamic kwargs into the evaluator
        run_evaluation(args.model, args.task, **kwargs)
    except Exception as e:
        print(f"[Error] Pipeline failed: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()