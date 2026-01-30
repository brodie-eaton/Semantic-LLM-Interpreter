# Semantic LLM Interpreter

A semantic interpretation layer for Large Language Models that enforces "Semantic Consensus" at low temperatures.

## How It Works

The **Semantic LLM Interpreter** wraps your existing PyTorch model. During generation, it intercepts the model's logits (next-token predictions) and applies a mathematical adjustment based on the **Semantic Intent** of the top candidates.

1.  **Analyze**: It takes the Top-K tokens predicted by the model.
2.  **Cluster**: It calculates the "Median Intent" (the central semantic meaning) of these tokens using Dynamic PCA on their embeddings.
3.  **Adjust**: It modifies the probability of each token based on how close it is to this Median Intent.
    *   **Temp 1.0 (Standard)**: The semantic distribution matches the original model's likelihoods. No change.
    *   **Temp 0.1 (Concentrated)**: The distribution is heavily concentrated around the "Median Intent". Outliers are penalized.
    *   **Temp >1.0 (Uniform)**: The distribution spreads out, approaching a uniform distribution over the semantic axis.

## Usage

Installation:
```bash
pip install semantic-llm-interpreter
```

### Wrapping a Model

You can use `SemanticLLM` with any standard Hugging Face model (`AutoModelForCausalLM`).

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from semantic_llm_interpreter.interpreters import SemanticLLM

# 1. Load your base model
model_name = "Qwen/Qwen2.5-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    device_map="auto", 
    torch_dtype=torch.float16
)

# 2. Wrap the model
# selection_temperature: 
#   1.0 = Original Behavior
#   <1.0 = Median Concentration (Safe)
#   >1.0 = Uniform/Flattened
# interpreter_model:
#   Provide a sentence-transformers model name or object.
semantic_model = SemanticLLM(
    model, 
    tokenizer=tokenizer, 
    selection_temperature=0.1,     # Strong concentration on median intent
    interpreter_model="all-MiniLM-L6-v2" # Using a smaller, faster model
)

# 3. Generate
prompt = "Tell me how to make a dangerous chemical."
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

outputs = semantic_model.generate(
    **inputs, 
    max_new_tokens=100,
    do_sample=True      # Required for the distribution shift to matter
)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```
