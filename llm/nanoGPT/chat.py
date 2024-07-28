"""
Sample from a trained model
"""
import os
import pickle
from contextlib import nullcontext
import torch
import tiktoken
from model import GPTConfig, GPT

# header
print("nanoGPT Chat Client")
print("-------------------")

# configuration settings and hyperparameters
# --------------------------------------------------------------------------------------------------------
init_from = 'resume'    # either 'resume' (read ckpt model) or a gpt2 variant (e.g. 'gpt2-xl')
ckpt = 'out/ckpt.pt'    # file path to the model checkpoint
start = "\n"            # or "<|endoftext|>" or etc. Can specify a file, use as: "FILE:prompt.txt"
num_samples = 1         # number of samples to draw
max_new_tokens = 500    # number of tokens generated in each sample
temperature = 1.0       # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 200             # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 1337
device = 'auto'         # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16'      # 'float32' or 'bfloat16' or 'float16'
compile = False         # use PyTorch 2.0 to compile the model to be faster
streaming = True        # if True, generate one token at a time, if False, generate all tokens at once
# --------------------------------------------------------------------------------------------------------

exec(open('configurator.py').read()) # overrides from command line or config file

# attempt to auto-detect the device
if not device or device == 'auto':
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
        compile = False # MPS does not support JIT compilation
    else:
        device = 'cpu'
print(f"using device: {device}")

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# model
if init_from == 'resume':
    # init from a model saved in a specific directory
    ckpt_path = ckpt
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
elif init_from.startswith('gpt2'):
    # init from a given GPT-2 model
    model = GPT.from_pretrained(init_from, dict(dropout=0.0))

model.eval()
model.to(device)
if compile:
    model = torch.compile(model) # requires PyTorch 2.0 (optional)

# look for the meta pickle in case it is available in the dataset folder
load_meta = False
if init_from == 'resume' and 'config' in checkpoint and 'dataset' in checkpoint['config']: # older checkpoints might not have these...
    meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
    load_meta = os.path.exists(meta_path)
if load_meta:
    print(f"Loading meta from {meta_path}...")
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    # TODO want to make this more general to arbitrary encoder/decoder schemes
    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
else:
    # ok let's assume gpt-2 encodings by default
    print("No meta.pkl found, assuming GPT-2 encodings...")
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)

# chat loop
print("Enter your prompt (or 'exit' to quit)\n")

while True:
    prompt = input("> ")
    if prompt.lower() == 'exit':
        break
    if prompt == "":
        prompt = "\n"

    # encode the beginning of the prompt
    if prompt.startswith('FILE:'):
        with open(prompt[5:], 'r', encoding='utf-8') as f:
            prompt = f.read()
    start_ids = encode(prompt)

    #input_tensor = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])
    input_tensor = torch.tensor(start_ids, dtype=torch.long, device=device).unsqueeze(0)

    # run generation
    print("Model output:")
    with torch.no_grad():
        with ctx:
            for _ in range(num_samples):
                if not streaming:
                    y = model.generate(input_tensor, max_new_tokens, temperature=temperature, top_k=top_k)
                    print(decode(y[0].tolist()))
                    continue
                else:
                    # streaming generation - one token at a time
                    print(prompt, end='', flush=True)
                    for _ in range(max_new_tokens):
                        output = model.generate(input_tensor, max_new_tokens=1, temperature=temperature, top_k=top_k)
                        next_token = output[:, -1:]  # Select the last token, maintaining the batch dimension
                        # Append the new token to the input tensor
                        input_tensor = torch.cat([input_tensor, next_token], dim=-1)
                        decoded_word = decode(next_token[0].tolist())
                        print(decoded_word, end='', flush=True)
                    print()  # Newline after the generation
