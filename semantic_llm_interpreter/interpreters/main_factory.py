from semantic_llm_interpreter.interpreters.torch_interpreter import SemanticTorchInterpreter
# from semantic_llm_interpreter.interpreters.keras_interpreter import SemanticKerasInterpreter
from semantic_llm_interpreter.interpreters.adapters import HAS_TORCH, HAS_TF

def SemanticLLM(model, tokenizer=None, 
                selection_temperature=None, 
                interpreter_model="all-mpnet-base-v2",
                max_context_length=4096):
    """
    Factory function for the Semantic LLM Interpreter.
    
    This wraps a standard LLM (Torch) to enforce Semantic Consensus during generation.
    
    Args:
        model: The model object (PyTorch nn.Module).
        tokenizer: The tokenizer (required for decoding candidates).
        selection_temperature: (Default 0.1) Target Semantic Std Dev. 
                               1.0 = Normal (Original), <1.0 = Median, >1.0 = Uniform.
        interpreter_model: (Default "all-mpnet-base-v2") The specific embedding model to use for interpretation.
                           Can be a string (HF Model Name) or a custom object with `.encode()`.
        max_context_length: (Default 4096) Max characters of context to process for interpretation.
        
    Returns:
        A wrapped model object with the same interface.
    """
    # Detect Type
    is_torch = False
    if HAS_TORCH:
        import torch
        if isinstance(model, torch.nn.Module):
            is_torch = True
            
    is_keras = False
    if HAS_TF:
        import tensorflow as tf
        if isinstance(model, tf.keras.Model):
            is_keras = True
            
    if is_torch:
        return SemanticTorchInterpreter(model, tokenizer, 
                                    selection_temperature=selection_temperature, 
                                    interpreter_model=interpreter_model,
                                    max_context_length=max_context_length)
        
    if is_keras:
        # Placeholder for Keras support update
        # return SemanticKerasInterpreter(model, tokenizer)
        raise NotImplementedError("Keras support is temporarily disabled pending refactor.")
        
    raise ValueError(f"Unknown model type: {type(model)}. Expected PyTorch nn.Module.")
