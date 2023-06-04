#!/usr/bin/python3
"""
Use the OpenAI API

Author: Jason A. Cox
3 June 2023
https://github.com/jasonacox/ProtosAI

This is a test of the OpenAI API. It requires that you register for an API
key at https://platform.openai.com/

"""

import openai

# Set up your OpenAI API key at openai.com
openai.api_key = "{insert your API key here}"

gpt_prompt = input ("Question: ") 

message=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": gpt_prompt},
]
response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages = message,
    temperature=0.2,
    max_tokens=1000,
    frequency_penalty=0.0
)
print(f"Answer: {response}")

content = response["choices"][0]["message"]["content"]

print(f"\n{content}")
