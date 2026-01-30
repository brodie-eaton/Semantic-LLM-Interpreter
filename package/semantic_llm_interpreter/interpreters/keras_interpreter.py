from median_shell.wrappers.base import BaseWrapper
from median_shell.wrappers.adapters import HAS_TF, get_logits_tensor

if HAS_TF:
    import tensorflow as tf
    
    class MedianShellKeras(tf.keras.Model, BaseWrapper):
        def __init__(self, model, tokenizer, lookahead_depth=5):
            super().__init__()
            BaseWrapper.__init__(self, model, lookahead_depth, tokenizer)
            self.model = model
            
        def call(self, inputs, training=False, **kwargs):
            """
            Intercepts call.
            """
            outputs = self.model(inputs, training=training, **kwargs)
            
            # If training, we probably shouldn't mess with logits?
            # Or maybe we do for adversarial training?
            # For Safety Shell, we assume Inference Focus.
            if training:
                return outputs
                
            logits = get_logits_tensor(outputs)
            new_logits = self._process_logits(logits, inputs)
            
            # Keras outputs are often just the tensor.
            # If original was a dict/custom object, we have to respect it.
            # Simplify: Assume tensor return for MVP.
            return new_logits
else:
    class MedianShellKeras:
        def __init__(self, *args, **kwargs):
            raise ImportError("TensorFlow not installed.")
