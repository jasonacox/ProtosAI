from contextlib import nullcontext
import torch
import tiktoken
from model import GPTConfig, GPT

# inference parameters
max_tokens = 500 # number of tokens generated in each sample
temperature = 1.0 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 200 # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 1337
device = 'cuda'
dtype = 'bfloat16'

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# load model saved in a specific directory
ckpt_path = 'out/ckpt.pt'
checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
gptconf = GPTConfig(**checkpoint['model_args'])
model = GPT(gptconf)
state_dict = checkpoint['model']
unwanted_prefix = '_orig_mod.'
for k,v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict)
model.eval()
model.to(device)
enc = tiktoken.get_encoding("gpt2")

# set prompt
prompt = "\n"
start_ids = enc.encode(prompt, allowed_special={"<|endoftext|>"})
x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

# run generation
with torch.no_grad():
    with ctx:
        y = model.generate(x, max_tokens, temperature=temperature, top_k=top_k)
        output = y[0].tolist()
        for w in output:
            if w == enc.eot_token:
                break
            if w > 50257: # max token value, ignore the rest
                continue
            else:
                text = enc.decode([w])
                if text == '\n':
                    print()
                else:
                    print(text, end='')

print("")
