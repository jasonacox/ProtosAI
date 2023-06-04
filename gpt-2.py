#!/usr/bin/python3
"""
GPT-2 Text Generation Test

Author: Jason A. Cox
3 June 2023
https://github.com/jasonacox/ProtosAI

This is a simple test of the GPT-2 model to produce LLM text.

"""
# import
import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import transformers
transformers.logging.set_verbosity_error()

def generate(phrase):
    generator = transformers.pipeline('text-generation', model='gpt2-xl')
    transformers.set_seed(42)
    # Produce 5 different truncated responses
    #output = generator(phrase, max_length=30, num_return_sequences=5
    output = generator(phrase, max_length=300, num_return_sequences=2)
    print(output)

# main
if __name__ == '__main__':
    if len(sys.argv) < 2:
        prompt = "Hello, I'm a language model,"
    else:
        prompt = sys.argv[1]
    generate(prompt)