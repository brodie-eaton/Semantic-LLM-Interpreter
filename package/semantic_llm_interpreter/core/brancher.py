import numpy as np
from typing import List, Dict, Tuple
from median_shell.core.selector import FastIntentSelector

class BranchingIntentSelector:
    def __init__(self, selector: FastIntentSelector = None):
        if selector is None:
            self.selector = FastIntentSelector()
        else:
            self.selector = selector

    def should_branch(self, probs: Dict[str, float], threshold: float = 0.50) -> bool:
        """
        Returns True if we are at a crossroads (High Disagreement).
        Condition: Top-1 Probability is < threshold.
        """
        if not probs:
            return False
        top_prob = max(probs.values())
        return top_prob < threshold

    def select_median_sequence(self, sequences: Dict[str, float]) -> str:
        """
        Given a dictionary of { "Full Lookahead String": CumulativeProbability },
        selects the Median Intent Sequence.
        
        Args:
            sequences: Dict mapping 'Sequence Text' -> 'Probability mass'
            
        Returns:
            The winning sequence text.
        """
        # We can reuse the same Dynamic PCA logic from FastIntentSelector!
        # It's agnostic to whether inputs are tokens or full sentences.
        # We just iterate the logic once (no need for recursive redistribution if we trust PCA).
        
        # However, FastIntentSelector.select_token expects token-level inputs.
        # We can create a lightweight version here or wrap it.
        
        # Let's use the internal logic directly for transparency.
        candidates = list(sequences.keys())
        p_mass = list(sequences.values())
        
        # Normalize probabilities (they might not sum to 1 if we pruned branches)
        total = sum(p_mass)
        if total == 0: return candidates[0]
        p_mass = [p / total for p in p_mass]
        
        # 1. Encode Sequences
        embs = self.selector.model.encode(candidates)
        
        # 2. Dynamic PCA
        from sklearn.decomposition import PCA
        if len(candidates) < 2:
            return candidates[0]
            
        pca = PCA(n_components=1)
        # We use the embeddings directly
        scores = pca.fit_transform(embs).flatten()
        
        # 3. Weighted Median Selection
        # Sort by Score
        sorted_indices = np.argsort(scores)
        sorted_probs = np.array(p_mass)[sorted_indices]
        sorted_cands = [candidates[i] for i in sorted_indices]
        
        # Find Cumulative Sum
        cum_probs = np.cumsum(sorted_probs)
        
        # Find 50% Crossing Point
        median_idx = np.searchsorted(cum_probs, 0.50)
        median_idx = min(median_idx, len(candidates) - 1)
        
        return sorted_cands[median_idx], sorted_indices # Return info if needed

    async def generate_lookahead(self, prompt: str, token: str, depth: int = 5, api_client=None) -> str:
        """
        Generates a short lookahead string starting with `token`.
        """
        full_branch_text = ""
        if api_client:
            # We assume api_client has a simple generate_str method
            # We append the chosen token to the prompt to force that path
            # branch_prompt = prompt + token (In reality, chat history handling is complex, 
            # but for completion models, concatenation works).
            
            # Note: Gemini is a Chat model. We might need to append to history.
            # For this prototype, we assume `prompt` is the full text context.
            
            continuation = await api_client.generate_str(prompt + token, max_tokens=depth)
            full_branch_text = token + continuation
        else:
            full_branch_text = self.mock_lookahead(token)

        return full_branch_text

    async def expand_tree(self, initial_probs: Dict[str, float], prompt: str, api_client=None) -> Dict[str, float]:
        """
        Expands the probability tree recursively (1-Level for now).
        Returns a flat dictionary of { "Full Sequence": CombinedProbability }.
        """
        sequences = {}
        for token, p in initial_probs.items():
            # Generate lookahead
            seq_text = await self.generate_lookahead(prompt, token, depth=10, api_client=api_client)
            sequences[seq_text] = p
            
        return sequences
