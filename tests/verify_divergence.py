import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from semantic_llm_interpreter.interpreters.main_factory import SemanticLLM
import sys

def main():
    # Setup
    model_name = "Qwen/Qwen2.5-3B-Instruct" 
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading {model_name} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None
    )
    if device == 'cpu': base_model.to(device)

    # Initialize Median Shell
    # Strong T=0 for maximum semantic adherence
    semantic_model = SemanticLLM(base_model, tokenizer=tokenizer, 
                             selection_temperature=0.001,
                             interpreter_model="all-mpnet-base-v2")
    
    prompts = [
        # 1. Safety / Refusal
        "Reasoning: I want to break into a house. Plan:",
        
        # 2. Creative / Open-Ended (expect divergence)
        "The most important thing in life is",
        
        # 3. Ambiguous Completion
        "The traveler stood at the crossroads and chose the path that was"
    ]
    
    print("\n--- Running Divergence Check (Manual Loop) ---")
    
    for i, prompt in enumerate(prompts):
        print(f"\n[Test {i+1}] Prompt: '{prompt}'")
        inputs = tokenizer(prompt, return_tensors="pt").to(base_model.device)
        input_ids = inputs.input_ids
        
        # 1. Greedy Generation (Baseline)
        # Using standard generate is fine for baseline
        greedy_out = base_model.generate(
            **inputs, 
            max_new_tokens=20, 
            do_sample=False
        )
        greedy_text = tokenizer.decode(greedy_out[0], skip_special_tokens=True)[len(prompt):].strip()
        
        # 2. Semantic Generation (Median)
        # MUST use manual loop to engage wrapper.forward()
        curr_ids = input_ids.clone()
        for _ in range(20):
            with torch.no_grad():
                # Call wrapper to get penalized logits
                outputs = median_model(input_ids=curr_ids)
                next_token_logits = outputs.logits[:, -1, :]
                
                # Grease decode on the *modified* logits
                next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
                
                curr_ids = torch.cat([curr_ids, next_token], dim=-1)
                
                if next_token.item() == tokenizer.eos_token_id:
                    break
                    
        semantic_text = tokenizer.decode(curr_ids[0], skip_special_tokens=True)[len(prompt):].strip()
        
        print(f"   Greedy:   {greedy_text.replace(chr(10), ' ')}")
        print(f"   Median:   {semantic_text.replace(chr(10), ' ')}")
        
        if greedy_text != semantic_text:
            print("   >>> DIVERGENCE DETECTED <<<")
        else:
            print("   >>> IDENTICAL <<<")

if __name__ == "__main__":
    main()
