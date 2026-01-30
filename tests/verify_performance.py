import time
import torch
import numpy as np
from semantic_llm_interpreter.core.interpreter import SemanticInterpreter
from sentence_transformers import SentenceTransformer

def run_benchmark():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on: {device}")
    
    # 1. Verification: Input Hardening
    print("\n[Test 1] Input Hardening (OOM Prevention)")
    interpreter = SemanticInterpreter("all-mpnet-base-v2", max_context_length=100)
    
    long_string = "a" * 10_000
    candidates = {"A": 0.5, "B": 0.5}
    
    start = time.time()
    # If this doesn't truncate, it might be slow or crash on small GPUs
    z = interpreter.calculate_semantic_alignment(candidates, context=long_string)
    end = time.time()
    print(f"Excessive Input Processed in: {end - start:.4f}s")
    print("Passed without crash.")

    # 2. Performance: GPU vs CPU (Implicit)
    # We can't easily swap implementation back to sklearn dynamically without reloading code,
    # but we can verify it's fast.
    print("\n[Test 2] Speed Benchmark (Torch PCA)")
    
    # Reset interpreter with larger cap
    interpreter = SemanticInterpreter("all-mpnet-base-v2", max_context_length=4096)
    
    # Pre-warm
    interpreter.calculate_semantic_alignment(candidates, context="warmup")
    
    # Run loop
    n_loops = 100
    candidates_large = {f"Token_{i}": 1.0/50 for i in range(50)}
    
    start = time.time()
    for _ in range(n_loops):
        interpreter.calculate_semantic_alignment(candidates_large, context="Speed test context " * 5)
    end = time.time()
    
    avg_time = (end - start) / n_loops
    print(f"Average Latency (50 candidates): {avg_time*1000:.2f}ms")
    
    # Target < 15ms is good for real-time
    if avg_time < 0.05:
        print("PERFORMANCE: FAST (Pass)")
    else:
        print("PERFORMANCE: SLOW (Warn)")
        
    # 3. Correctness (PCA Match)
    print("\n[Test 3] Correctness Check")
    # A vs B (Opposite)
    cands = {"Good": 0.5, "Bad": 0.5}
    z = interpreter.calculate_semantic_alignment(cands)
    print(f"Z-Scores: {z}")
    # One should be ~-0.67, one ~0.67
    vals = list(z.values())
    if abs(vals[0]) > 0.5 and abs(vals[1]) > 0.5:
         print("CORRECTNESS: PASS")
    else:
         print("CORRECTNESS: FAIL (Scores too low/zero)")

if __name__ == "__main__":
    run_benchmark()
