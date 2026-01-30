
import pytest
import torch
from semantic_llm_interpreter.interpreters.main_factory import SemanticLLM

class MockModel(torch.nn.Module):
    pass
class MockTokenizer:
    pass

def test_temperature_ranges():
    # T=0 (Should clamp)
    s0 = SemanticLLM(MockModel(), MockTokenizer(), selection_temperature=0.0)
    assert s0.temperature == 0.0 # Stored as 0, logic clamps at runtime
    
    # T=1 (Normal)
    s1 = SemanticLLM(MockModel(), MockTokenizer(), selection_temperature=1.0)
    
    # T=100 (High)
    s100 = SemanticLLM(MockModel(), MockTokenizer(), selection_temperature=100.0)
    
    # T=-1 (Negative - should clamp or handle)
    s_neg = SemanticLLM(MockModel(), MockTokenizer(), selection_temperature=-1.0)

def test_interpreter_types():
    # String
    SemanticLLM(MockModel(), MockTokenizer(), interpreter_model="all-MiniLM-L6-v2")
    
    # Object
    class MyObj:
        def encode(self, x): return []
    SemanticLLM(MockModel(), MockTokenizer(), interpreter_model=MyObj())
    
    # Invalid
    with pytest.raises(ValueError):
        SemanticLLM(MockModel(), MockTokenizer(), interpreter_model=None)
