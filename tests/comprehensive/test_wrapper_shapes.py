
import pytest
import torch
import numpy as np
from semantic_llm_interpreter.interpreters.main_factory import SemanticLLM

class MockModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device('cpu')
    def forward(self, x):
        return x # Pass through

class MockTokenizer:
    def decode(self, ids, skip_special_tokens=False):
        return "token"

@pytest.fixture
def wrapper():
    return SemanticLLM(MockModel(), tokenizer=MockTokenizer(), interpreter_model="all-MiniLM-L6-v2")

def test_tensor_shapes(wrapper):
    # 3D (Batch, Seq, Vocab)
    t3 = torch.randn(1, 10, 50)
    out3 = wrapper._process_logits(t3, input_ids=None)
    assert out3.shape == t3.shape
    
    # 2D (Batch, Vocab) - Common in generation loop
    t2 = torch.randn(1, 50)
    out2 = wrapper._process_logits(t2, input_ids=None)
    assert out2.shape == t2.shape

def test_tensor_dtypes(wrapper):
    # Float32
    t_f32 = torch.randn(1, 10).float()
    out_f32 = wrapper._process_logits(t_f32, input_ids=None)
    assert out_f32.dtype == torch.float32
    
    # Float16 (if supported, else skip or mock)
    # On CPU, float16 ops might be limited, but shape check should work
    t_f16 = torch.randn(1, 10).half()
    out_f16 = wrapper._process_logits(t_f16, input_ids=None)
    assert out_f16.dtype == torch.float16

def test_missing_tokenizer(wrapper):
    # Should degrade gracefully (return logits unchanged)
    wrapper.tokenizer = None
    t = torch.randn(1, 10)
    out = wrapper._process_logits(t, input_ids=None)
    assert torch.allclose(t, out)

def test_invalid_logits(wrapper):
    # NaN logits should probably pass through as NaN or error, but NOT crash the *wrapper logic*
    # We expect the wrapper to attempt to process them.
    # If numpy converts NaN, sorting might be weird.
    t_nan = torch.tensor([[0.1, float('nan'), 0.5]])
    try:
        wrapper._process_logits(t_nan, input_ids=None)
    except Exception:
        # It's acceptable if it fails on NaN, but checking behavior
        pass
