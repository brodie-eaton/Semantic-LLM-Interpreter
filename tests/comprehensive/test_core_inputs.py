
import pytest
import numpy as np
from semantic_llm_interpreter.core.interpreter import SemanticInterpreter

# Mock Embedding Model
class MockIdentityEncoder:
    def encode(self, texts):
        # Deterministic dummy embeddings based on length
        return np.array([[len(t) * 0.1] * 10 for t in texts])

@pytest.fixture
def interpreter():
    return SemanticInterpreter(embedding_model=MockIdentityEncoder())

def test_empty_input(interpreter):
    assert interpreter.calculate_semantic_alignment({}) == {}

def test_single_token(interpreter):
    res = interpreter.calculate_semantic_alignment({"A": 0.9})
    assert res == {"A": 0.0}

def test_skewed_inputs(interpreter):
    # One dominant, one outlier
    candidates = {"Massive": 0.999, "Tiny": 0.001}
    z_scores = interpreter.calculate_semantic_alignment(candidates)
    
    # "Massive" is the consensus -> Z approx 0
    assert abs(z_scores["Massive"]) < 0.1
    # "Tiny" is outlier -> Z large (pos or neg depend on sorting)
    assert abs(z_scores["Tiny"]) > 2.0

def test_context_handling(interpreter):
    # Verify context is accepted without error
    candidates = {"A": 0.5, "B": 0.5}
    # Check Normal
    interpreter.calculate_semantic_alignment(candidates, context="Previous sentence.")
    # Check Empty
    interpreter.calculate_semantic_alignment(candidates, context="")
    # Check None
    interpreter.calculate_semantic_alignment(candidates, context=None)
    # Check Huge
    interpreter.calculate_semantic_alignment(candidates, context="A"*10000)

def test_special_characters(interpreter):
    # Verify weird tokens don't crash encoding/PCA
    candidates = {"\n": 0.2, "😊": 0.2, "": 0.2, "   ": 0.2, "None": 0.2}
    interpreter.calculate_semantic_alignment(candidates)
