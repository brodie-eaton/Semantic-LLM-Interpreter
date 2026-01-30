import numpy as np
import pytest
from semantic_llm_interpreter.core.interpreter import SemanticInterpreter

class MockInterpreter:
    def encode(self, texts):
        # Return dummy embeddings
        # We need 1D PC1 to effectively just be the sorted order of these texts
        # So we can just map them to scalar values
        # Let's say: "A" < "B" < "C" < "D"
        mapping = {"A": -2.0, "B": -1.0, "C": 1.0, "D": 2.0}
        
        # Output shape must be (N, D). Let's use D=2
        # We'll make the first dimension the significant one
        vectors = []
        for t in texts:
            val = mapping.get(t, 0.0)
            vectors.append([val, 0.0])
        return np.array(vectors)

def test_quantile_mapping_balanced():
    """Test 50/50 split."""
    selector = SemanticInterpreter(embedding_model=MockInterpreter())
    
    # A vs D (Opposite ends)
    candidates = {"A": 0.5, "D": 0.5}
    
    # PC1 Order: A (-2), D (2).
    # Sorted: A, D
    # Probs: 0.5, 0.5
    # CumSum: 0.5, 1.0
    # Midpoints: 0.25, 0.75
    # Z-Scores: norm.ppf(0.25) ~ -0.67, norm.ppf(0.75) ~ 0.67
    
    z_scores = selector.calculate_semantic_alignment(candidates)
    
    print(f"Balanced Z-Scores: {z_scores}")
    
    assert "A" in z_scores
    assert "D" in z_scores
    assert np.isclose(z_scores["A"], -0.674, atol=0.01)
    assert np.isclose(z_scores["D"], 0.674, atol=0.01)

def test_quantile_mapping_skewed():
    """Test skewed distribution (Median Shift)."""
    selector = SemanticInterpreter(embedding_model=MockInterpreter())
    
    # A (10%), B (10%), C (10%), D (70%)
    # D is the massive mode.
    # Order: A, B, C, D
    # Probs: 0.1, 0.1, 0.1, 0.7
    # CumSum: 0.1, 0.2, 0.3, 1.0
    # Midpoints: 0.05, 0.15, 0.25, 0.65
    
    # Z-Scores:
    # A (0.05) -> -1.645
    # B (0.15) -> -1.036
    # C (0.25) -> -0.674
    # D (0.65) -> +0.385 (Median is roughly here)
    
    candidates = {"A": 0.1, "B": 0.1, "C": 0.1, "D": 0.7}
    z_scores = selector.calculate_semantic_alignment(candidates)
    
    print(f"Skewed Z-Scores: {z_scores}")
    
    assert z_scores["A"] < z_scores["B"] < z_scores["C"] < z_scores["D"]
    assert np.isclose(z_scores["D"], 0.385, atol=0.01)

if __name__ == "__main__":
    # Manual run
    try:
        test_quantile_mapping_balanced()
        test_quantile_mapping_skewed()
        print("All tests passed!")
    except Exception as e:
        print(f"Test failed: {e}")
        raise
