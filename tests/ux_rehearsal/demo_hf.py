
# UX Start-to-Finish Simulation
# Goal: Load a model, wrap it, and generate text.

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from semantic_llm_interpreter import SemanticLLM

# 1. Load Standard Model (Small for speed/compatibility in test)
model_name = "Qwen/Qwen2.5-0.5B-Instruct" 
print(f"Loading {model_name}...")

try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto")
except Exception as e:
    print(f"Failed to load model: {e}")
    print("Ensure you have internet access and HF_TOKEN if needed.")
    exit(1)

# 2. Wrap Model
print("\nWrapping model with SemanticLLM...")
# Friction Point Check: Does this generally Just Work?
try:
    # Explicitly using a smaller embedding model for speed in this test
    # but normally users might use default mpnet
    wrapped_model = SemanticLLM(model, tokenizer, 
                                selection_temperature=0.1, 
                                interpreter_model="all-MiniLM-L6-v2")
except Exception as e:
    print(f"Wrapper Failed: {e}")
    exit(1)

# 3. Generate
print("\nGenerating...")
prompt = "The capital of France is"
inputs = tokenizer(prompt, return_tensors="pt")

# Standard Generation call
# Friction check: Does wrapped_model.generate behave exactly like model.generate?
try:
    outputs = wrapped_model.generate(**inputs, max_new_tokens=20)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\nOutput: {text}")
    print("\nSUCCESS: End-to-End flow complete.")
except Exception as e:
    print(f"Generation Failed: {e}")
    exit(1)
