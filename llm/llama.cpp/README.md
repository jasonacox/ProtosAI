# Using LLaMA Models

The [llama.cpp project's](https://github.com/ggerganov/llama.cpp) goal is to run LLaMA models using 4-bit integer quantization on small scale computers like a MacBook.

## Setup

```bash
# Clone the project
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp

# Linux - Build for Nvidia GPU using CMake
mkdir build
cd build
cmake .. -DLLAMA_CUBLAS=ON   # Omit -DLLAMA_CUBLAS=ON for CPU only
cmake --build . --config Release

# MacOS - Build using Metal
# or Linux - CPU only
make -j

# MacOS - Build without Metal
LLAMA_NO_METAL=1 make -j
```

## Use Pre-Trained Models

```bash
# Download the LLaMA-2 7B GGUF model from Hugging Face.
cd models
# Also llama-2-7b-chat.Q3_K_M.gguf for small 3 bit version
wget https://huggingface.co/TheBloke/Llama-2-7b-Chat-GGUF/resolve/main/llama-2-7b-chat.Q5_K_M.gguf
cd ..

# Run interactive chat.
./build/bin/main -m models/llama-2-7b-chat.Q5_K_M.gguf \
    -t 4 \
    --color \
    -c 4096 \
    --temp 0.7 \
    --gpu-layers 32 \
    -n -1 \
    -i -ins 

Where:
    -m models/llama-2-7b-chat.Q5_K_M.gguf   # The model
    -t 4                                    # change to match number of CPU cores
    -c 4096                                 # context length
    --temp 0.7                              # randomness 
    --gpu-layers 32                         # number of layers to offload to GPU - remove if cpu only
    -n -1 --color                           # options 
    -i -ins                                 # interactive mode and instruction
```

See https://github.com/ggerganov/llama.cpp/blob/master/examples/main/README.md for more details on inference parameters.

Example chat using llama.cpp interactive mode:

```bash
./main -t 4 -m models/llama-2-7b-chat.Q5_K_M.gguf \
    --color -c 4096 --temp 0.7 --gpu-layers 32 -n -1 -i -ins
```

```
> Pick a color
Green

> What is a llama?
A llama is a large, domesticated mammal that is native to South America. It has long, shaggy fur and distinctive ears with a distinctive curled-over shape. Llamas are used for their wool and as pack animals in remote areas where cars cannot reach. They are also known for their calm and gentle nature.

> Write a haiku
Llama in the sun
Gentle eyes, shaggy coat
Soft as a cloud
```

## Python Interface

The models built or downloaded here can be used by the [LLaMa-cpp-python](https://github.com/abetlen/llama-cpp-python) project.

```bash
# MacOS - Install python module
pip install llama-cpp-python

# Linux OS with Nvidia GPU support
CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install llama-cpp-python
```

This will also build llama.cpp similar to what we did in in Setup above. Next, if you downloaded and converted the open_llama_3b model above, you can test it using this python script:

```python
from llama_cpp import Llama

# Load model
llm = Llama(model_path="models/llama-2-7b-chat.Q5_K_M.gguf")

# Ask a question
question = "Name the planets in the solar system?"
print(f"Asking: {question}...")
output = llm(f"Q: {question} A: ", 
    max_tokens=64, stop=["Q:", "\n"], echo=True)

# Print answer
print("\nResponse:")
print(output['choices'][0]['text'])
```

## OpenAI API Compatible Server

The llama-cpp-python library has a built in OpenAI API compatible server. This can be used to host your model locally and use OpenAI API tools against your self-hosted LLM.

```bash
# Install Server that uses OpenAI API
CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install llama-cpp-python[server]

# Run the API Server
python3 -m llama_cpp.server \
    --model ./models/llama-2-7b-chat.Q5_K_M.gguf \
    --host 10.0.1.89 \
    --n_gpu_layers 20 

# It will listen on port 8000
```

See the example [chat.py](chat.py) CLI Chatbot script that connects to this server and hosts
an interactive session with the LLM.

The example chat.py Features:
  * Use of OpenAI API (could be used to connect to the OpenAI service if you have a key)
  * Works with local hosted OpenAI compatible llama-cpp-python[server]
  * Retains conversational context for LLM
  * Uses response stream to render LLM chunks instead of waiting for full response

## Train

The llama.cpp project includes a `train-text-from-scratch` tool. Use `-h` to see the options or an example below.

```bash
# Create a text file to use for training
mkdir models/jason
curl -s https://github.com/jasonacox/ProtosAI/files/11715802/input.txt > models/jason/jason.txt

# Run the training
./train-text-from-scratch \
        --vocab-model models/ggml-vocab.bin \
        --checkpoint-in models/jason/jason.chk.bin \
        --checkpoint-out models/jason/jason.chk.bin \
        --model-out models/jason/jason.bin \
        --train-data models/jason/jason.txt \
        --ctx 32 --embd 256 --head 8 --layer 16 \
        -t 4 -b 32 --seed 42 --adam-iter 16 \
        --use-flash --print-details-interval 0 --predict 64 \
        -n 1 # adjust this for the number of iterations to run
```

## Chat

The `main` tool includes options for prompts and interactivity. There are example scripts in the `examples` folder of the project. I created my own [ai.txt](ai.txt) prompt and used this script to generate a chat:

```bash
#!/bin/bash
./main -m models/openlm-research_open_llama_3b/ggml-model-f16.bin \
    -c 512 -b 1024 -n 256 --keep 48 \
	--repeat_penalty 1.0 --color -i \
	-r "User:" \
	-f prompts/ai.txt

```
Example session:

```
== Running in interactive mode. ==
 - Press Ctrl+C to interject at any time.
 - Press Return to return control to LLaMa.
 - To return control without starting a new line, end your input with '/'.
 - If you want to submit another line, end your input with '\'.

 Transcript of a dialog, where the User interacts with an Assistant named AI. AI is helpful, kind, honest, good at writing, and never fails to answer the User's requests immediately and with precision.

User:Hello, AI.
AI:Hello. How may I help you today?
User:Please tell me the largest city in Europe.
AI:Sure. The largest city in Europe is Moscow, the capital of Russia.
User:What are the primary colors?
AI:The primary colors are red, blue, and yellow.
User:How many months are in a year?
AI:There are 12 months in a year.
User:What are the 9 planets in our solar system?
AI:The planets in our solar system are: Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, Neptune, and Pluto.
User:
```

## References

* LLaMa.cpp - https://github.com/ggerganov/llama.cpp
* LLaMa-cpp-python - https://github.com/abetlen/llama-cpp-python
* Video Tutorial - Train your own llama.cpp mini-ggml-model from scratch!: https://asciinema.org/a/592303
* How to run your own LLM GPT - https://blog.rfox.eu/en/Programming/How_to_run_your_own_LLM_GPT.html

## Additional Tools

* LlamaIndex - augment LLMs with our own private data - https://gpt-index.readthedocs.io/en/latest/index.html

