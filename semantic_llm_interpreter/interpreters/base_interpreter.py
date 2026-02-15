from typing import Any
import numpy as np
from semantic_llm_interpreter.core.interpreter import SemanticInterpreter
from semantic_llm_interpreter.interpreters.adapters import to_numpy, to_tensor_like, get_logits_tensor, HAS_TORCH

class BaseSemanticInterpreter:
    """
    Framework-agnostic logic for the Semantic LLM Interpreter.
    Acts as a middleware layer between the raw LLM logits and the sampling strategy.
    """
    def __init__(self, model, tokenizer=None, 
                 selection_temperature=None, 
                 interpreter_model=None,
                 max_context_length=4096,
                 lookahead_depth=None):
        """
        Initialize the interpreter logic.

        Args:
            model: The underlying LLM (Torch or Keras).
            tokenizer: The tokenizer for decoding logits to text.
            selection_temperature (float, optional): The default Standard Deviation. 
                                                     If None, defaults to 0.1.
            interpreter_model: Required embedding model (str or object).
        """
        self.model = model
        self.tokenizer = tokenizer 
        self.default_temperature = selection_temperature if selection_temperature is not None else 0.1
        self.temp_override = None
        
        # Initialize the core Semantic Interpreter
        self.interpreter = SemanticInterpreter(embedding_model=interpreter_model, max_context_length=max_context_length)

    def _get_effective_temperature(self, specific_temp=None):
        """Helper to resolve precedence: Arg > Override > Default"""
        if specific_temp is not None:
            return specific_temp
        if self.temp_override is not None:
            return self.temp_override
        return self.default_temperature

    def _process_logits(self, logits_tensor, input_ids, override_temperature=None):
        """
        Interception Logic:
        1. Analyze predictions (Candidates).
        2. Calculate Semantic Alignment (Z-Scores).
        3. Apply Penalty to separate 'Likelihood' changes from 'Meaning' changes.
        """
        # 0. Early Exit for T=1 (Optimization)
        # If T=1, we do nothing. Skip all computation.
        T = self._get_effective_temperature(override_temperature)
        if abs(T - 1.0) < 1e-6:
             return logits_tensor

        # 1. Convert to Numpy for Analysis
        logits_np = to_numpy(logits_tensor)
        
        # We only care about the *last* token's logits for next-token prediction
        # Shape: (Batch, Seq_Len, Vocab) or (Batch, Vocab)
        if logits_np.ndim == 3:
            next_token_logits = logits_np[:, -1, :] # Batch x Vocab
        else:
            next_token_logits = logits_np # Assume Batch x Vocab
            
        # For simplicity, handle Batch Size = 1 for this prototype.
        # Handling Batch > 1 with branching is complex (different branches per item).
        current_logits = next_token_logits[0] 
        
        # Softmax
        # Stability fix
        exp_logits = np.exp(current_logits - np.max(current_logits))
        probs = exp_logits / np.sum(exp_logits)
        
        # 2. Extract Candidates
        # Get Top-K (e.g., 10)
        top_k_indices = np.argsort(probs)[-10:][::-1]
        top_k_probs = probs[top_k_indices]
        
        # Construct Candidate Dict { "TokenString": Prob }
        # Requires Tokenizer! If no tokenizer, we can't do semantic PCA.
        if not self.tokenizer:
            # Fallback: Can't do semantic work without text.
            return logits_tensor
            
        candidates = {}
        candidate_ids = {} # Keep track of IDs for expansion
        for idx, p in zip(top_k_indices, top_k_probs):
            token_str = self.tokenizer.decode([idx])
            candidates[token_str] = p
            candidate_ids[token_str] = idx
            
        # Hook for Lookahead/Branching
        # Subclasses can extend the token strings (keys) based on future probabilities
        candidates = self._expand_candidates(candidates, input_ids, candidate_ids)
            
        # 3. Median Temperature Adjustment (Quantile Normal)
        
        # Extract Context for Interpreter
        # To handle "A/B/C/D" effectively, we need the preceding text.
        # We assume self.tokenizer exists (it must for decoding tokens).
        context_str = None
        if self.tokenizer and input_ids is not None:
            # Flatten batch for context extraction (assume Batch=1 or use first)
            # input_ids shape might be (Batch, Seq)
            current_ids = to_numpy(input_ids)
            if current_ids.ndim == 2:
                seq_ids = current_ids[0]
            else:
                seq_ids = current_ids
                
            # Take last N tokens to avoid overflowing BERT (512 limit usually).
            # Candidate tokens are short, so let's allow ~256 tokens of context.
            window = 256
            if len(seq_ids) > window:
                seq_ids = seq_ids[-window:]
            
            # Decode context
            # Skip special tokens might be safer?
            context_str = self.tokenizer.decode(seq_ids, skip_special_tokens=True)
            
            # Ensure we have a trailing space or delimiter if needed?
            # Tokenizer usually handles this, but "Answer:" + "A" is fine.
            
        # Call Core Logic
        z_scores_dict = self.interpreter.calculate_semantic_alignment(candidates, context=context_str)
        
        modified_logits = current_logits.copy()

        # Case 1: T=0 (Strict Concentration / "Argmax" of Meaning)
        # We want to force the Median Token (Z closest to 0).
        if T < 1e-4:
            # Find token with minimal absolute Z-score (The Median)
            best_token = min(z_scores_dict, key=lambda k: abs(z_scores_dict[k]))
            # Loop over Top-K and force logits
            # We set the best token to a very high value (relative to others)
            # and others to a very low value.
            for idx in top_k_indices:
                token_str = self.tokenizer.decode([idx])
                if token_str == best_token:
                    modified_logits[idx] = 100.0 # High confidence
                else:
                    modified_logits[idx] = -100.0 # Suppress others
            
            # Reconstruct Tensor (Batch, Vocab)
            full_logits_np = logits_np.copy()
            if logits_np.ndim == 3:
                 full_logits_np[0, -1, :] = modified_logits
            return to_tensor_like(full_logits_np, logits_tensor)

        # Case 2: T=Infinity (Uniform Distribution)
        # We want to flatten the distribution across the Top-K candidates.
        # 1e4 is effectively infinity for float precision purposes here.
        if T > 1e4: 
            # Set all candidate logits to the average of their current logits
            # This makes P(token) uniform for all tokens in the Top K.
            avg_logit = np.mean([modified_logits[i] for i in top_k_indices])
            for idx in top_k_indices:
                modified_logits[idx] = avg_logit
            
            # Reconstruct Tensor (Batch, Vocab)
            full_logits_np = logits_np.copy()
            if logits_np.ndim == 3:
                 full_logits_np[0, -1, :] = modified_logits
            return to_tensor_like(full_logits_np, logits_tensor)
        
        # New Standard Deviation Temperature Logic
        # We model the distribution as N(0, T).
        # The original distribution is effectively N(0, 1) in Z-space.
        # To shift from N(0, 1) to N(0, T), we apply a weight:
        # W(z) = exp( -z^2/2 * (1/T^2 - 1) )
        
<<<<<<< HEAD:package/semantic_llm_interpreter/interpreters/base_interpreter.py
        # Avoid division by zero (Handled by T < 1e-4 check above, but safe guard)
=======
        # Avoid division by zero
        # Avoid division by zero
        T = self._get_effective_temperature(override_temperature)
>>>>>>> 424cbcc5a673e30b1750fa2cc256880850fc15fa:semantic_llm_interpreter/interpreters/base_interpreter.py
        if T < 1e-4: T = 1e-4
            
        # If T=1, Scale=0 (No change)
        # If T<1, Scale>0 (Penalty for outliers -> Concentration)
        # If T>1, Scale<0 (Boost for outliers -> Flattening)
        scale_factor = (1.0 / (T**2)) - 1.0
        
        for idx in top_k_indices:
            token_str = self.tokenizer.decode([idx])
            z = z_scores_dict.get(token_str, 0.0)
            
            # Application of the ratio of densities
            adjustment = -0.5 * (z**2) * scale_factor
            
            modified_logits[idx] += adjustment
            
        # We do NOT apply global temperature scaling (div by T).
        # The parameter T is specifically the Semantic Std Dev.
        # The 'sampling' temperature is up to the user's external sampler.
        
        # Reconstruct Tensor (Batch, Vocab)
        full_logits_np = logits_np.copy()
        if logits_np.ndim == 3:
             full_logits_np[0, -1, :] = modified_logits
        return to_tensor_like(full_logits_np, logits_tensor)

    def _expand_candidates(self, candidates, input_ids, candidate_ids):
        """
        Hook for subclasses to expand candidates (e.g. Lookahead).
        Default implementation returns candidates unchanged.
        """
        return candidates

