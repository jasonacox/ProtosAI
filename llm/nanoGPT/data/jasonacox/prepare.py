#!/usr/bin/python3
"""
Tokenize all content from inputfile

Author: Jason A. Cox
14 June 2023
https://github.com/jasonacox/ProtosAI/

Credit: Simon Willison
    * Modified code from https://til.simonwillison.net/llms/training-nanogpt-on-my-blog  
"""
import os
import json
import tiktoken
import numpy as np
import random

# input file
input_file_path = os.path.join(os.path.dirname(__file__), 'input.nl-json')

# array of blog posts
entries = []

# read posts, one per line
print(f"Reading blog posts from {input_file_path}...")
n = 1
with open(input_file_path, 'r', encoding='utf-8') as f:
    for line in f:
        if line.strip():
            entries.append(json.loads(line))
        print(f"Post {n}\r",end="")
        n = n + 1

# shuffle entries
print("\nShuffle posts...")
random.shuffle(entries)

# create training set and validation set
print("Creating training and validation sets...")
n = len(entries)
train_entries = entries[:int(n*0.9)]
val_entries = entries[int(n*0.9):]

# tokenizer
enc = tiktoken.get_encoding("gpt2")

# loop through entries, tonize, add EOS token
print("Tokenizing training set...")
train_ids = []
for entry in train_entries:
    content = f"{entry[0]}\nby Jason A. Cox\n\n{entry[1]}"
    train_ids += enc.encode_ordinary(content)
    train_ids.append(enc.eot_token)
print("Tokenizing validation set...")
val_ids = []
for entry in val_entries:
    content = f"{entry[0]}\nby Jason A. Cox\n\n{entry[1]}"
    val_ids += enc.encode_ordinary(content)
    val_ids.append(enc.eot_token)

# print stats
print(f" - train has {len(train_ids):,} tokens")
print(f" - val has {len(val_ids):,} tokens")

# export to bin files
print("Writing to disk...")
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))
print("Done. Output files written: train.bin and val.bin")
