# Using LLaMA Models

The [llama.cpp project's](https://github.com/ggerganov/llama.cpp) goal is to run LLaMA models using 4-bit integer quantization on small scale computers like a MacBook.

## Setup

```bash
# Clone the project
https://github.com/ggerganov/llama.cpp.git
cd llama.cpp

# Build for local machine
make
```

## Use Pre-Trained Models

```bash
# Download the Open LLaMA 3B, 7B, or 13B model from Hugging Face.
cd models
git-lfs clone https://huggingface.co/openlm-research/open_llama_3b # or use 
cd ..
python convert.py models/openlm-research/open_llama_3b

# Test
./main -m models/openlm-research/open_llama_3b
```

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

The `main` tool includes options for prompts and interactivity. There are example scripts in the `examples` folder of the project.

## Python Interface

The models built or downloaded here can be used by the [LLaMa-cpp-python](https://github.com/abetlen/llama-cpp-python) project.

```bash
# Install python module
pip install llama-cpp-python
```

This will also build llama.cpp similar to what we did in in Setup above. Next, if you downloaded and converted the open_llama_3b model above, you can test it using this python script:

```python
from llama_cpp import Llama

# Load model
llm = Llama(model_path="models/openlm-research_open_llama_3b/ggml-model-f16.bin")

# Ask a question
question = "Name the planets in the solar system?"
print(f"Asking: {question}...")
output = llm(f"Q: {question} A: ", 
    max_tokens=64, stop=["Q:", "\n"], echo=True)

# Print answer
print("\nResponse:")
print(output['choices'][0]['text'])
```

Example Run

```bash
llama.cpp: loading model from models/openlm-research_open_llama_3b/ggml-model-f16.bin
llama_model_load_internal: format     = ggjt v1 (pre #1405)
llama_model_load_internal: n_vocab    = 32000
llama_model_load_internal: n_ctx      = 512
llama_model_load_internal: n_embd     = 3200
llama_model_load_internal: n_mult     = 240
llama_model_load_internal: n_head     = 32
llama_model_load_internal: n_layer    = 26
llama_model_load_internal: n_rot      = 100
llama_model_load_internal: ftype      = 1 (mostly F16)
llama_model_load_internal: n_ff       = 8640
llama_model_load_internal: n_parts    = 1
llama_model_load_internal: model size = 3B
llama_model_load_internal: ggml ctx size =    0.06 MB
llama_model_load_internal: mem required  = 7559.86 MB (+  682.00 MB per state)
llama_new_context_with_model: kv self size  =  162.50 MB
AVX = 1 | AVX2 = 1 | AVX512 = 0 | AVX512_VBMI = 0 | AVX512_VNNI = 0 | FMA = 1 | NEON = 0 | ARM_FMA = 0 | F16C = 1 | FP16_VA = 0 | WASM_SIMD = 0 | BLAS = 1 | SSE3 = 1 | VSX = 0 | 
Asking: Name the planets in the solar system?...

llama_print_timings:        load time =  6330.48 ms
llama_print_timings:      sample time =    54.73 ms /    36 runs   (    1.52 ms per token,   657.77 tokens per second)
llama_print_timings: prompt eval time =  6330.38 ms /    14 tokens (  452.17 ms per token,     2.21 tokens per second)
llama_print_timings:        eval time = 14147.08 ms /    35 runs   (  404.20 ms per token,     2.47 tokens per second)
llama_print_timings:       total time = 20687.86 ms

Response:
Q: Name the planets in the solar system? A: 1. Mercury 2. Venus 3. Earth 4. Mars 5. Jupiter 6. Saturn 7. Uranus 8. Neptune
```

## References

* LLaMa.cpp - https://github.com/ggerganov/llama.cpp
* LLaMa-cpp-python - https://github.com/abetlen/llama-cpp-python
* Video Tutorial - Train your own llama.cpp mini-ggml-model from scratch!: https://asciinema.org/a/592303
* How to run your own LLM GPT - https://blog.rfox.eu/en/Programming/How_to_run_your_own_LLM_GPT.html

## Additional Tools

* LlamaIndex - augment LLMs with our own private data - https://gpt-index.readthedocs.io/en/latest/index.html

