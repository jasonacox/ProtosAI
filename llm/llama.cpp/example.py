#!/usr/bin/python3
"""
Llama_cpp Prompt Example

Python script to load Llama model and prompt it with
a question.

Author: Jason A. Cox
8 July 2023
https://github.com/jasonacox/ProtosAI

"""
import sys
from llama_cpp import Llama

# Load model - see https://github.com/jasonacox/ProtosAI/tree/master/llm/llama.cpp#setup
llm = Llama(model_path="models/openlm-research_open_llama_3b/ggml-model-f16.bin")

# Ask a question
question = "Name the planets in the solar system?"
print(f"Asking: {question}...")
output = llm(f"Q: {question} A: ", 
    max_tokens=64, stop=["Q:", "\n"], echo=True)

# Print answer
print("\nResponse:")
print(output['choices'][0]['text'])
