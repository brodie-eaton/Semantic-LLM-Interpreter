import numpy as np

# Conditional Imports to support environments with partial constraints
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import tensorflow as tf
    HAS_TF = True
except ImportError:
    HAS_TF = False

def to_numpy(tensor):
    """
    Converts a PyTorch/TensorFlow tensor or List to a numpy array.
    """
    if isinstance(tensor, np.ndarray):
        return tensor
    
    if HAS_TORCH and isinstance(tensor, torch.Tensor):
        if tensor.dtype == torch.bfloat16:
             return tensor.float().detach().cpu().numpy()
        return tensor.detach().cpu().numpy()
        
    if HAS_TF and tf.is_tensor(tensor):
        return tensor.numpy()
        
    if isinstance(tensor, list):
        return np.array(tensor)
        
    raise TypeError(f"Unsupported tensor type: {type(tensor)}")

def to_tensor_like(array, reference):
    """
    Converts a numpy array back to the type and device of the reference tensor.
    """
    if HAS_TORCH and isinstance(reference, torch.Tensor):
        t = torch.from_numpy(array)
        t = t.to(reference.device, dtype=reference.dtype)
        return t
        
    if HAS_TF and tf.is_tensor(reference):
        return tf.convert_to_tensor(array, dtype=reference.dtype)
        
    return array

def get_logits_tensor(output):
    """
    Extracts the logits tensor from model output (which might be a tuple, dict, or class).
    Common HF pattern: ModelOutput.logits or (logits, ...) tuple.
    """
    if HAS_TORCH and isinstance(output, torch.Tensor):
        return output
    if HAS_TF and tf.is_tensor(output):
        return output
        
    # Handle HF ModelOutput or Tuples
    if hasattr(output, 'logits'):
        return output.logits
    if isinstance(output, (list, tuple)):
        return output[0] # Assume logits are first
        
    return output
