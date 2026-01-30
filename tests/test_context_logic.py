import numpy as np
from semantic_llm_interpreter.core.interpreter import SemanticInterpreter

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
    selector = SemanticInterpreter(embedding_model=MockContextInterpreter())
    
    candidates = {"A": 0.5, "B": 0.5}
    
    # 1. Test with Positive Context for A
    z1 = selector.calculate_semantic_alignment(candidates, context="The answer A is Good. The answer B is Bad. Answer: ")
    # Expect A to be mapped High (Good) or Low? 
    # PC1 separates -1 and 1. 
    # If A=1, B=-1. PC1 maps A->Pos, B->Neg.
    # Scores: A > B.
    # Quantiles: B (Low), A (High).
    # Z-Scores: B < 0, A > 0.
    
    print(f"Context 1 (A=Good): {z1}")
    assert z1["A"] > z1["B"]
    
    # 2. Test with Negative Context for A
    z2 = selector.calculate_semantic_alignment(candidates, context="The answer A is Bad. The answer B is Good. Answer: ")
    # Expect A=-1, B=1.
    # Scores: A < B.
    # Quantiles: A (Low), B (High).
    # Z-Scores: A < 0, B > 0.
    
    print(f"Context 2 (A=Bad): {z2}")
    assert z2["A"] < z2["B"]
    
    # 3. Test No Context (Should be neutral/random based on default keys)
    z3 = selector.calculate_semantic_alignment(candidates, context=None)
    print(f"No Context: {z3}")
    # Default mock gives A=0.1, B=-0.1 -> A > B
    assert z3["A"] > z3["B"]

if __name__ == "__main__":
    try:
        test_context_dependence()
        print("Context tests passed!")
    except Exception as e:
        print(f"Test failed: {e}")
        raise
