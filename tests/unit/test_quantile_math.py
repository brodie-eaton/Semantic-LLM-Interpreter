import pytest
import numpy as np
from scipy.stats import norm
from semantic_llm_interpreter.core.interpreter import SemanticInterpreter

# We can re-use the selector fixture or test strict math
# Since the logic is inside check `calculate_similarities`, we test that.

class IdentityInterpreter:
    """Returns vectors where [0] is the value determining order."""
    def encode(self, texts):
        # We start texts with numbers to control order: "1_A", "2_B"
        vecs = []
        for t in texts:
            val = float(t.split('_')[0])
            v = np.zeros(10)
            v[0] = val
            vecs.append(v)
        return np.array(vecs)

@pytest.fixture
def math_interpreter():
    return SemanticInterpreter(embedding_model=IdentityInterpreter())

def test_quantile_math_balanced(math_interpreter):
    """
    Test 50/50 split. 
    A (val=1) -> 0.5 mass
    B (val=2) -> 0.5 mass
    
    PC1 should align with value.
    Ordered: A, B.
    CumSum: 0.5, 1.0.
    Midpoints: 0.25, 0.75.
    Z-Scores: norm.ppf(0.25) ~= -0.67, norm.ppf(0.75) ~= +0.67
    """
    candidates = {"1_A": 0.5, "2_B": 0.5}
    z = math_interpreter.calculate_semantic_alignment(candidates)
    
    # A is "left", B is "right" (or vice versa depending on PCA sign flip)
    # But absolute values should be close to 0.67
    
    z_a = z["1_A"]
    z_b = z["2_B"]
    
    assert np.isclose(abs(z_a), 0.67, atol=0.05)
    assert np.isclose(abs(z_b), 0.67, atol=0.05)
    
    # They should have opposite signs
    assert np.sign(z_a) != np.sign(z_b)

def test_quantile_math_skewed(math_interpreter):
    """
    Test 10/90 split.
    A (val=1) -> 0.1
    B (val=2) -> 0.9
    
    Ordered: A, B
    Cum: 0.1, 1.0
    Mid: 0.05, 0.55
    Z(A) = ppf(0.05) ~= -1.64
    Z(B) = ppf(0.55) ~= 0.12
    """
    candidates = {"1_A": 0.1, "2_B": 0.9}
    z = math_interpreter.calculate_semantic_alignment(candidates)
    
    z_a = z["1_A"]
    z_b = z["2_B"]
    
    assert np.isclose(abs(z_a), 1.64, atol=0.1)
    assert np.isclose(abs(z_b), 0.12, atol=0.1)

def test_quantile_clipping(math_interpreter):
    """Test extremely skewed inputs to ensure we clip and don't return inf."""
    candidates = {"1_A": 1e-9, "2_B": 1.0 - 1e-9}
    z = math_interpreter.calculate_semantic_alignment(candidates)
    
    # norm.ppf(0) is -inf. We clip to epsilon.
    assert not np.isinf(z["1_A"])
    assert not np.isinf(z["2_B"])
