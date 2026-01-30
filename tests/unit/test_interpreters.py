import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock
from semantic_llm_interpreter.interpreters.torch_interpreter import SemanticTorchInterpreter

# Mock Model
class MockGPT(nn.Module):
    def __init__(self, vocab_size=100, hidden_size=32):
        super().__init__()
        self.vocab_size = vocab_size
        self.head = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, input_ids):
        # Return deterministic logits based on input
        batch, seq = input_ids.shape
        # Use sum of input_ids to seed or offset
        seed = input_ids.sum().item()
        torch.manual_seed(seed)
        
        logits = torch.randn(batch, seq, self.vocab_size)
        
        # Cast to head's dtype to simulate real model behavior
        return logits.to(self.head.weight.dtype)

# Mock Tokenizer
class MockTokenizer:
    def decode(self, ids, skip_special_tokens=True):
        return "".join([chr(65 + i % 26) for i in ids]) # A, B, C...

# Mock Interpreter
class MockInterpreter:
    def encode(self, texts):
        return torch.randn(len(texts), 16).numpy()

@pytest.fixture
def interpreter_instance():
    base_model = MockGPT()
    tokenizer = MockTokenizer()
    interpreter = MockInterpreter()
    # T=0.1 -> Strong Concentration
    return SemanticTorchInterpreter(base_model, tokenizer, 
                                selection_temperature=0.1, 
                                interpreter_model=interpreter)

def test_forward_shape_preservation(interpreter_instance):
    """Output logits should match input shape."""
    input_ids = torch.randint(0, 100, (2, 10)) # Batch 2, Seq 10
    outputs = interpreter_instance(input_ids)
    
    assert outputs.shape == (2, 10, 100)
    assert isinstance(outputs, torch.Tensor)

def test_forward_bfloat16_compatibility(interpreter_instance):
    """Regression Test: BFloat16 logits should not crash."""
    input_ids = torch.randint(0, 100, (1, 5))
    
    # Force model to produce bf16
    interpreter_instance.model.to(dtype=torch.bfloat16)
    interpreter_instance.model.head.weight.data = interpreter_instance.model.head.weight.data.bfloat16()
    interpreter_instance.model.head.bias.data = interpreter_instance.model.head.bias.data.bfloat16()
    
    # Run
    # Note: CPU might not support bf16 fully, usually requires MKLDNN or just works slow.
    try:
        outputs = interpreter_instance(input_ids)
        # Should return float32 (we verified we cast to float for numpy) 
        # OR return BF16 if we cast back. The wrapper currently returns float32 or whatever `to_tensor_like` does.
        # `to_tensor_like` uses the source tensor's type.
        assert outputs.dtype == torch.bfloat16
    except RuntimeError as e:
        if "not implemented" in str(e):
            pytest.skip("BFloat16 not supported on this CPU hardware")
        else:
            raise e

def test_forward_modifies_logits(interpreter_instance):
    """Ensure that median logic actually changes the logits."""
    input_ids = torch.randint(0, 100, (1, 5))
    
    # 1. Run raw model
    with torch.no_grad():
        raw_logits = interpreter_instance.model(input_ids)
        
    # 2. Run wrapper
    with torch.no_grad():
        wrapped_logits = interpreter_instance(input_ids)
        
    # Last token logits should differ
    raw_last = raw_logits[0, -1, :]
    wrapped_last = wrapped_logits[0, -1, :]
    
    # Check max difference
    diff = torch.abs(raw_last - wrapped_last).max().item()
    assert diff > 0.001, "Wrapper did not modify logits!"

def test_no_tokenizer_passthrough(interpreter_instance):
    """If tokenizer is None, should passthrough exactly."""
    interpreter_instance.tokenizer = None
    input_ids = torch.randint(0, 100, (1, 5))
    
    with torch.no_grad():
        raw = interpreter_instance.model(input_ids)
        wrapped = interpreter_instance(input_ids)
        
    assert torch.allclose(raw, wrapped)
