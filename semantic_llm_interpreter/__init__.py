from semantic_llm_interpreter.core.interpreter import SemanticInterpreter

# Interpreters
# We attempt to expose SemanticLLM factory.
try:
    from semantic_llm_interpreter.interpreters.main_factory import SemanticLLM
except ImportError:
    pass
