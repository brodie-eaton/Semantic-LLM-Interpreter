import pytest
import torch
from semantic_llm_interpreter import SemanticLLM
from semantic_llm_interpreter.interpreters.torch_interpreter import SemanticTorchInterpreter

# Reuse Mocks
class MockGPT(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.head = torch.nn.Linear(32, 100)
    def forward(self, input_ids):
        return torch.randn(input_ids.shape[0], input_ids.shape[1], 100)

class MockTokenizer:
    def decode(self, ids, skip_special_tokens=True):
        return "TEST"

def test_factory_creation():
    """Test the top-level SemanticLLM factory."""
    model = MockGPT()
    tokenizer = MockTokenizer()
    
    # Should detect torch model and return SemanticTorchInterpreter
    interpreter_obj = SemanticLLM(model, tokenizer=tokenizer)
    
    assert isinstance(interpreter_obj, SemanticTorchInterpreter)
    assert interpreter_obj.tokenizer is tokenizer

def test_manual_generation_loop_integration():
    """Simulate a localized generation loop as used in benchmarks."""
    model = MockGPT()
    tokenizer = MockTokenizer()
    interpreter = SemanticLLM(model, tokenizer=tokenizer, selection_temperature=0.1)
    
    input_ids = torch.randint(0, 100, (1, 5))
    
    # Run 5 steps
    for _ in range(5):
        outputs = interpreter(input_ids) # Should return modified logits
        # In integration, we care that it returns a Tensor we can use
        logits = outputs
        next_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(-1)
        input_ids = torch.cat([input_ids, next_token], dim=-1)
        
    assert input_ids.shape == (1, 10)

def test_interpreter_injection_integration():
    """Ensure custom interpreter is passed down correctly."""
    model = MockGPT()
    tokenizer = MockTokenizer()
    
    # Mock Interpreter
    class CustomInterpreter:
        def encode(self, texts):
            return torch.zeros(len(texts), 16).numpy()
            
    interpreter_model = CustomInterpreter()
    
    interpreter = SemanticLLM(model, tokenizer=tokenizer, interpreter_model=interpreter_model)
    
    # Check deeper
    # interpreter (SemanticTorchInterpreter) -> base (BaseSemanticInterpreter) -> interpreter (SemanticInterpreter) -> model
    assert interpreter.interpreter.model is interpreter_model
