from setuptools import setup, find_packages

setup(
    name="semantic_llm_interpreter",
    version="0.1.0",
    packages=['semantic_llm_interpreter', 'semantic_llm_interpreter.core', 'semantic_llm_interpreter.interpreters'],
    install_requires=[
        "sentence-transformers",
        "numpy",
        "torch"
    ],
)
