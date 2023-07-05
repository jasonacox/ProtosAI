#
# XGen
# Official research release for the family of XGen models (7B) by Salesforce AI Research:
#
# Title: Long Sequence Modeling with XGen: A 7B LLM Trained on 8K Input Sequence Length
#
# Authors: 
#    Erik Nijkamp*, Tian Xie*, Hiroaki Hayashi*, Bo Pang*, Congying Xia*, Chen Xing, 
#    Rui Meng, Wojciech Kryscinski, Lifu Tu, Meghana Bhat, Semih Yavuz, Jesse Vig, 
#    Lidiya Murakhovs'ka, Chien-Sheng Wu, Yingbo Zhou, Shafiq Rayhan Joty, Caiming Xiong, 
#    Silvio Savarese.
#
# Model cards are published on the HuggingFace Hub:
#
# XGen-7B-4K-Base with support for 4K sequence length.
# XGen-7B-8K-Base with support for 8K sequence length.
# XGen-7B-8k-Inst with instruction-finetuning (for research purpose only).
#
# Repo: https://github.com/salesforce/xgen 
#

# import
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

print("load tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("Salesforce/xgen-7b-8k-base", trust_remote_code=True)

print("load model...")
model = AutoModelForCausalLM.from_pretrained("Salesforce/xgen-7b-8k-base", torch_dtype=torch.bfloat16)

print("generating...")
inputs = tokenizer("The world is", return_tensors="pt")
sample = model.generate(**inputs, max_length=128)
print(tokenizer.decode(sample[0]))
