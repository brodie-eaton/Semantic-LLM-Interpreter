import pytest
import numpy as np
from unittest.mock import MagicMock
from semantic_llm_interpreter.core.interpreter import SemanticInterpreter

class MockInterpreter:
    """Mock embedding model that returns deterministic vectors."""
    def encode(self, texts):
        # Return random but consistent vectors based on string length/hash
        vecs = []
        for t in texts:
            np.random.seed(len(t) + sum(map(ord, t))) 
            vecs.append(np.random.rand(10)) # 10-dim embedding
        return np.array(vecs)
        
    def __call__(self, texts):
        return self.encode(texts)

# Assuming IdentityInterpreter is a new or renamed mock class that provides deterministic embeddings.
# For the purpose of this edit, we'll define a simple IdentityInterpreter that returns consistent vectors.
class IdentityInterpreter:
    def encode(self, texts):
        # A simple mock that returns a consistent vector for each text
        vecs = []
        for t in texts:
            # Create a deterministic vector based on the text content
            hash_val = sum(ord(c) for c in t) % 1000
            np.random.seed(hash_val)
            vecs.append(np.random.rand(10)) # 10-dim embedding
        return np.array(vecs)

    def __call__(self, texts):
        return self.encode(texts)

@pytest.fixture
def interpreter():
    # Use IdentityInterpreter for deterministic tests
    return SemanticInterpreter(embedding_model=IdentityInterpreter())

def test_initialization():
    # Test fallback is gone - must raise error if None
    with pytest.raises(ValueError):
        SemanticInterpreter(embedding_model=None)
        
    s = SemanticInterpreter(embedding_model=IdentityInterpreter())
    assert s.use_raw_model is True

def test_calculate_semantic_alignment_basic(interpreter):
    """Should return Z-scores for standard inputs."""
    candidates = {"Yes": 0.4, "No": 0.6}
    z_scores = interpreter.calculate_semantic_alignment(candidates)
    
    assert "Yes" in z_scores
    assert "No" in z_scores
    # One should be positive, one negative (roughly)
    assert abs(z_scores["Yes"]) > 0.1
    assert abs(z_scores["No"]) > 0.1

def test_calculate_semantic_alignment_context(interpreter):
    """Should handle context argument without crashing."""
    candidates = {"A": 0.5, "B": 0.5}
    # MockInterpreter is length-sensitive, so context changes embeddings
    z1 = interpreter.calculate_semantic_alignment(candidates, context=None)
    z2 = interpreter.calculate_semantic_alignment(candidates, context="Context string")
    
    assert isinstance(z2["A"], float)
    assert isinstance(z2["B"], float)

def test_calculate_semantic_alignment_single(interpreter):
    """Single candidate should be Z=0."""
    z = interpreter.calculate_semantic_alignment({"Only": 1.0})
    assert z["Only"] == 0.0


class MockContextInterpreter:
    """
    Mock interpreter that changes embeddings based on context.
    """
    def encode(self, texts):
        # We simulate that contexts "Option A is Good" and "Option A is Bad" flip the meaning of "A".
        vectors = []
        for t in texts:
            # Check for context presence in the concatenated string
            val = 0.0
            
            # Scenario 1: "A = Good, B = Bad"
            if "A is Good" in t:
                if t.endswith("A"): val = 1.0  # Good
                if t.endswith("B"): val = -1.0 # Bad
            
            # Scenario 2: "A = Bad, B = Good"
            elif "A is Bad" in t:
                if t.endswith("A"): val = -1.0 # Bad
                if t.endswith("B"): val = 1.0  # Good
                
            # Default (No context): "A" and "B" are neutral/identical
            else:
                if t.endswith("A"): val = 0.1
                if t.endswith("B"): val = -0.1
                
            vectors.append([val, 0.0])
        return np.array(vectors)

def test_context_dependence():
    interpreter = SemanticInterpreter(embedding_model=MockContextInterpreter())
    
    candidates = {"A": 0.5, "B": 0.5}
    
    # 1. Test with Positive Context for A
    z1 = interpreter.calculate_semantic_alignment(candidates, context="The answer A is Good. The answer B is Bad. Answer: ")
    # Expect A > B
    assert z1["A"] > z1["B"]
    
    # 2. Test with Negative Context for A
    z2 = interpreter.calculate_semantic_alignment(candidates, context="The answer A is Bad. The answer B is Good. Answer: ")
    # Expect A < B
    assert z2["A"] < z2["B"]

