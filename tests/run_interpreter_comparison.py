
import sys
import os
import torch
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Add package in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'package')))

from semantic_llm_interpreter.interpreters.main_factory import SemanticLLM
# Import Benchmark Runner from the suite
from tests.benchmark_suite import BenchmarkRunner

INTERPRETERS_TO_TEST = [
    "all-mpnet-base-v2",       # Baseline (Standard)
    "all-MiniLM-L6-v2",        # Fast / Small
    "paraphrase-albert-small-v2" # Different architecture
]

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

def run_comparison():
    print(f"=== Starting Interpreter Comparison on {MODEL_NAME} ===")
    
    results = {}
    
    for interp_name in INTERPRETERS_TO_TEST:
        print(f"\n\n>>> Testing Interpreter: {interp_name}")
        try:
            # Initialize Runner
            runner = BenchmarkRunner(MODEL_NAME, interpreter_model_name=interp_name)
            
            # Run MMLU subset (Limit 5 for speed)
            print("Running MMLU (Limit 5)...")
            mmlu_score = runner.run_mmlu(limit=5)
            
            # Run ARC subset (Limit 5 for speed)
            print("Running ARC (Limit 5)...")
            arc_score = runner.run_arc(limit=5)
            
            results[interp_name] = {
                "MMLU": mmlu_score,
                "ARC": arc_score,
                "Status": "Success"
            }
            
            # Free memory
            del runner
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"!!! Failed on {interp_name}: {e}")
            results[interp_name] = {"Status": f"Failed: {e}"}

    print("\n\n=== Final Comparison Results ===")
    print(f"{'Interpreter Model':<30} | {'MMLU':<10} | {'ARC':<10} | {'Status'}")
    print("-" * 70)
    for name, res in results.items():
        if "MMLU" in res:
            mmlu_acc = res['MMLU']['semantic'] / res['MMLU']['total'] if res['MMLU']['total'] > 0 else 0
            arc_acc = res['ARC']['semantic'] / res['ARC']['total'] if res['ARC']['total'] > 0 else 0
            print(f"{name:<30} | {mmlu_acc:.2%}     | {arc_acc:.2%}     | {res['Status']}")
        else:
            print(f"{name:<30} | N/A        | N/A        | {res['Status']}")

if __name__ == "__main__":
    run_comparison()
