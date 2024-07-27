#!/usr/bin/python3
"""
Tokenize all content from TinyStories

Author: Jason A. Cox
25 July 2024
https://github.com/jasonacox/ProtosAI/

"""
import os
import tiktoken
import numpy as np
import random
import requests

# Grab TinyStories text file from Hugging Face
if not os.path.exists(os.path.join(os.path.dirname(__file__), 'tinystories.txt')):
    print("Downloading TinyStories from Hugging Face...")
    url = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-valid.txt"
    response = requests.get(url)
    input_file_path = os.path.join(os.path.dirname(__file__), 'tinystories.txt')
    with open(input_file_path, 'w', encoding='utf-8') as f:
        f.write(response.text)
    print("Done. TinyStories.txt written to disk.")
else:
    input_file_path = os.path.join(os.path.dirname(__file__), 'tinystories.txt')

# read stories into entries array
entries = []
print(f"Reading blog posts from {input_file_path}...")
n = 1
story = ""
with open(input_file_path, 'r', encoding='utf-8') as f:
    for line in f:
        if "<|endoftext|>" in line:
            # print(f"Story {n}: {story}\n")
            entries.append(story)
            story = ""
        else:
            story += line
        print(f"Story {n}\r",end="")
        n = n + 1

# shuffle entries
print("\nShuffle stories...")
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
for content in train_entries:
    train_ids += enc.encode_ordinary(content)
    train_ids.append(enc.eot_token)
print("Tokenizing validation set...")
val_ids = []
for content in val_entries:
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
