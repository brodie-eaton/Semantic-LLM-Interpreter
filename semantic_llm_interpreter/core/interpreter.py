import numpy as np
from sentence_transformers import SentenceTransformer
import logging

logger = logging.getLogger(__name__)


class SemanticInterpreter:
    """
    A core component that analyzes the 'meaning' of candidate tokens by projecting them
    into a semantic space and identifying the primary axis of intent (Principal Component).
    
    This allows the system to distinguish between surface-level probability (likelihood)
    and deep semantic alignment (intent).
    """
    def __init__(self, embedding_model, max_context_length=4096, logger=None):
        """
        Initialize the interpreter.
        
        Args:
            embedding_model: Either a string (HuggingFace model name) or an object with `.encode(texts)`.
                             This IS REQUIRED. The interpreter does not hold a built-in default.
            max_context_length (int): Maximum chacters of context to process. 
                                      Defaults to 4096 (approx 1000 tokens) to prevent DoS/OOM.
            logger (logging.Logger, optional): Custom logger instance.
        """
        self.logger = logger or logging.getLogger(__name__)
        self.max_context_length = max_context_length
        if embedding_model is None:
            raise ValueError("Relational/Embedding model is required. Please provide a `sentence-transformers` model name (str) or an object with an `.encode()` method.")
            
        if isinstance(embedding_model, str):
            self.logger.info(f"Loading semantic encoder: {embedding_model}...")
            self.model = SentenceTransformer(embedding_model)
            self.use_raw_model = False
        else:
            self.model = embedding_model
            self.use_raw_model = True
            
        self.logger.info("Semantic Interpreter Initialized.")
        
        # Load torch for PCA
        import torch
        self.torch = torch

    def calculate_semantic_alignment(self, candidates, context=None):
        """
        Analyzes a set of candidate tokens and computes their 'Semantic Alignment Score' (Z-Score).
        
        How it works:
        1. **Contextual Encoding**: Combines the prior context with each candidate to understand full meaning.
        2. **Manifold Discovery**: Uses PCA to find the single most important 'concept' that differentiates the candidates.
        3. **Quantile Normalization**: Maps the probabilistic weight of each token onto this semantic axis to find the 'Center of Meaning'.
        
        Args:
            candidates (dict): A dictionary mapping {token_string: probability}. 
                               Represents the model's top predictions.
            context (str, optional): The preceding text. Crucial for resolving ambiguity.
                                     
        Returns:
            dict: {token: alignment_score}. 
                  A score of 0.0 means the token represents the 'Median Intent'.
        """
        tokens = list(candidates.keys())
        if not tokens: return {}
        if len(tokens) == 1: return {tokens[0]: 0.0} 
        
        # 0. Input Hardening (DoS Prevention)
        if context and len(context) > self.max_context_length:
            context = context[-self.max_context_length:]
        
        # 1. Encode Candidates (with Context)
        if context:
            texts_to_encode = [context + t for t in tokens]
        else:
            texts_to_encode = tokens

        # Handle different model types (SentenceTransformer vs Raw Callable)
        # OPTIMIZATION: Keep on GPU if possible
        if hasattr(self.model, 'encode'):
            # SentenceTransformers supports convert_to_tensor
            try:
                embeddings = self.model.encode(texts_to_encode, convert_to_tensor=True)
            except TypeError:
                # Fallback for older versions or custom objects
                 embeddings = self.model.encode(texts_to_encode)
        else:
            out = self.model(texts_to_encode)
            embeddings = out # Assume tensor or handle later
            
        # Ensure tensor
        if not isinstance(embeddings, self.torch.Tensor):
            embeddings = self.torch.tensor(embeddings)
            
        # 2. Dynamic Semantic Axis (PCA via SVD)
        # We assume the candidates lie on a 1D spectrum of meaning.
        # PCA(1) is equivalent to the first singular vector of the centered data.
        
        # Center the data
        mean = embeddings.mean(dim=0, keepdim=True)
        centered = embeddings - mean
        
        # SVD (Fast on GPU)
        # U, S, Vh = torch.linalg.svd(centered, full_matrices=False)
        # PC1 direction is Vh[0] (right singular vector)
        # Projections = centered @ Vh[0].T
        
        # Note: torch.linalg.svd returns Vh (transposed V). 
        # So the first row of Vh is the first principal component.
        try:
            U, S, Vh = self.torch.linalg.svd(centered, full_matrices=False)
            
            # Deterministic Sign Flipping/Correction (mimic sklearn)
            # Ensure the first non-zero component of the first singular vector is positive
            # Vh[0] is the first principal component (eigenvector)
            components = Vh[0]
            max_abs_idx = self.torch.argmax(self.torch.abs(components))
            sign = self.torch.sign(components[max_abs_idx])
            
            # If 0, default to 1
            if sign == 0: sign = 1
            
            # Project onto PC1
            # centered: (N, D), Vh[0]: (D,) -> dot product
            # Apply sign correction so the projection is deterministic
            scores = self.torch.matmul(centered, Vh[0]) * sign
            
        except Exception as e:
            # Fallback for very small matrices or CPU edge cases
            self.logger.warning(f"SVD Failed: {e}, using CPU fallback")
            U, S, Vh = self.torch.linalg.svd(centered.cpu(), full_matrices=False)
            components = Vh[0]
            max_abs_idx = self.torch.argmax(self.torch.abs(components))
            sign = self.torch.sign(components[max_abs_idx])
            if sign == 0: sign = 1
            scores = self.torch.matmul(centered.cpu(), Vh[0]) * sign
            
        scores = scores.flatten().cpu().numpy() # Convert back to numpy for sorting/scipy logic
        
        probs = np.array(list(candidates.values()))
        
        # 3. Sort by Semantic Position
        # We organize tokens from "Left" to "Right" along the semantic axis.
        sorted_indices = np.argsort(scores)
        sorted_probs = probs[sorted_indices]
        sorted_tokens = [tokens[i] for i in sorted_indices]
        
        # 4. Calculate Cumulative Probability Mass (The "Vote")
        cum_probs = np.cumsum(sorted_probs)
        total_mass = cum_probs[-1]
        
        # Find the center of gravity (Median)
        # We calculate the midpoint of each token's probability block
        prev_cum = np.roll(cum_probs, 1)
        prev_cum[0] = 0.0
        midpoint_cdf = (prev_cum + cum_probs) / 2.0
        
        # Normalize to 0..1
        normalized_cdf = midpoint_cdf / total_mass
        
        # Clip to avoid infinity in Z-score calculation
        epsilon = 1e-4
        normalized_cdf = np.clip(normalized_cdf, epsilon, 1.0 - epsilon)
        
        # 5. Convert to Z-Scores (Standard Normal Deviations)
        # 0.0 = The exact median (Consensus)
        # +/- 1.0 = Standard deviations away from consensus
        from scipy.stats import norm
        z_scores = norm.ppf(normalized_cdf)
        
        return {t: float(z) for t, z in zip(sorted_tokens, z_scores)}
