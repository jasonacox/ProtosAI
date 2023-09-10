#!/usr/bin/python3
"""
Llama_cpp CLI ChatBot Example

Python script that uses the OpenAI API to provide
a command line interface (CLI) chat session with the LLM.

Features:
  * Uses OpenAI API
  * Works with local hosted OpenAI compatible llama-cpp-python[server]
  * Retains conversational context for LLM
  * Uses response stream to render LLM chunks instead of waiting for full response

Author: Jason A. Cox
10 Sept 2023
https://github.com/jasonacox/ProtosAI

"""
import openai
import datetime

# Configuration Settings - Showing local LLM
openai.api_key = "OPENAI_API_KEY"                # Required, use bogus string for Llama.cpp
openai.api_base = "http://localhost:8000/v1"     # Use API endpoint or comment out for OpenAI
agentname = "Jarvis"                             # Set the name of your bot
mymodel  ="./models/llama-2-7b-chat.Q5_K_M.gguf" # Pick model to use e.g. gpt-3.5-turbo for OpenAI
TESTMODE = False                                 # Uses test prompts

# Set base prompt and initialize the context array for conversation dialogue
current_date = datetime.datetime.now()
formatted_date = current_date.strftime("%m/%d/%Y")
baseprompt = "You are %s, a highly intelligent assistant. Keep your answers brief and accurate. Current date is %s." % (agentname, formatted_date)
context = [{"role": "system", "content": baseprompt}]

# Function - Send prompt to LLM for response
def ask(prompt):
    global context
    context.append({"role": "user", "content": prompt})
    #print(context)
    response = openai.ChatCompletion.create(
        model=mymodel,
        max_tokens=1024,
        stream=True, # Send response chunks as LLM computes next tokens
        temperature=0.7,
        messages=context,
    )
    return response

# Function - Render LLM response output in chunks
def printresponse(response):
    completion_text = ''
    # iterate through the stream of events and print it
    for event in response:
        event_text = event['choices'][0]['delta']
        if 'content' in event_text:
            chunk = event_text.content
            completion_text += chunk
            print(f"{chunk}",end="",flush=True) 
    print("",flush=True)
    return completion_text

# Chatbot Header
print(f"ChatBot - Greetings! My name is {agentname}. Enter an empty line to quit chat.")
print()

prompts = []
if TESTMODE:
    # define the series of questions here
    prompts.append("What is your name?")
    prompts.append("What is today's date?")
    prompts.append("What day of the week is it?")
    prompts.append("Answer this riddle: Ram's mom has three children, Reshma, Raja and a third one. What is the name of the third child?")
    prompts.append("Pick a color.")
    prompts.append("Now write a poem about that color.")
    prompts.append("What time is it?")
    prompts.append("Thank you very much!")

# Loop to prompt user for input
while True:
    if len(prompts) > 0:
        p = prompts.pop(0)
        print(f"> {p}")
    else:
        p = input("> ")
    if not p or p == "":
        break
    print()
    response=ask(p)
    print(f"{agentname}> ",end="", flush=True)
    ans = printresponse(response)
    context.append({"role": "assistant", "content" : ans})
    print()

print("Done")