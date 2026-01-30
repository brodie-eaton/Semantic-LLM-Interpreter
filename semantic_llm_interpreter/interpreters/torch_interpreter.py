import torch
import torch.nn as nn
from semantic_llm_interpreter.interpreters.base_interpreter import BaseSemanticInterpreter
from semantic_llm_interpreter.interpreters.adapters import get_logits_tensor

class SemanticTorchInterpreter(nn.Module, BaseSemanticInterpreter):
    """
    PyTorch-specific implementation of the Semantic LLM Interpreter.
    Wraps a huggingface/torch model to apply semantic control during the forward pass.
    """
    def __init__(self, model, tokenizer, 
                 selection_temperature=None, 
                 interpreter_model=None,
                 max_context_length=4096):
        """
        Args:
            model: PyTorch model (e.g. HuggingFace AutoModelForCausalLM).
            tokenizer: Tokenizer (e.g. HuggingFace AutoTokenizer).
            selection_temperature: (Default: 0.1) Target Semantic Std Dev.
            interpreter_model: Embedding model (str or object).
        """
        nn.Module.__init__(self)
        BaseSemanticInterpreter.__init__(self, model, tokenizer, 
                                     selection_temperature, 
                                     interpreter_model,
                                     max_context_length)
        self.model = model 

    def forward(self, *args, **kwargs):
        """
        Standard Wrapper Forward: Calls source, analyzes semantic intent, modifies logits, returns result.
        """
        # Run original source model
        outputs = self.model(*args, **kwargs)
        
        # If no tokenizer (no strings), we can't do anything
        if not self.tokenizer:
            return outputs

        # Extract Logits
        logits = get_logits_tensor(outputs)
        
        # We need input_ids
        input_ids = kwargs.get('input_ids', args[0] if args else None)
        
        # Check for dynamic temperature
        temp = kwargs.get('selection_temperature', None)
        
        new_logits = self._process_logits(logits, input_ids, override_temperature=temp)
        
        # Reconstruct Output
        if isinstance(outputs, tuple):
             return (new_logits,) + outputs[1:]
             
        if hasattr(outputs, 'logits'):
            outputs.logits = new_logits
            return outputs
            
        return new_logits

    def generate(self, *args, **kwargs):
        """
        Passthrough for generation. 
        Supports dynamic `selection_temperature`.
        """
        # Extract dynamic temperature if present
        temp = kwargs.pop('selection_temperature', None)
        
        # Set override state if provided
        if temp is not None:
            self.temp_override = temp
            
        try:
            return self.model.generate(*args, **kwargs)
        finally:
            # Always clean up override
            if temp is not None:
                self.temp_override = None
