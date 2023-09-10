import openai
import datetime

# For a locally hosted LLM
openai.api_key = "BOGUS_API_KEY"
openai.api_base = "http://x.x.x.x:8000/v1"
agentname = "Jarvis"
mymodel  ="./models/llama-2-7b-chat.Q5_K_M.gguf"

# The context array sets initial prompt and keeps track of conversation dialogue
current_date = datetime.datetime.now()
formatted_date = current_date.strftime("%m/%d/%Y")
context = [{"role": "system", "content": "You are %s, a highly intelligent assistant. Keep your answers brief and accurate. Current date is %s." % (agentname, formatted_date)}]

# Send prompt to LLM for response
def ask(prompt):
    global context
    context.append({"role": "user", "content": prompt})
    #print(context)
    response = openai.ChatCompletion.create(
        model=mymodel,
        max_tokens=1024,
        stream=True, # Send response chunks as LLM computes next tokens
        temp=0.7,
        messages=context,
    )
    return response

# Render LLM response output in chunks
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

# define the series of questions here
prompts = []
prompts.append("What is your name?")
prompts.append("What is today's date?")
prompts.append("What day of the week is it?")
prompts.append("Answer this riddle: Ram's mom has three children, Reshma, Raja and a third one. What is the name of the third child?")
prompts.append("Pick a color.")
prompts.append("Now write a poem about that color.")
prompts.append("What time is it?")
prompts.append("Thank you very much! Goodbye.")

for p in prompts:
    print(f"Prompt>\n{p}",flush=True)
    print()
    response=ask(p)
    print("Response>",flush=True)
    ans = printresponse(response)
    context.append({"role": "assistant", "content" : ans})
    print()

# Loop to prompt user for input
print(f"ChatBot - Greetings! My name is {agentname}. Enter an empty line to quit chat.")
print()
while True:
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