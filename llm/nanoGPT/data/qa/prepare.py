#!/usr/bin/python3
"""
Tokenize all content from questions.txt

Author: Jason A. Cox
3 Aug 2023
https://github.com/jasonacox/ProtosAI/

"""
import os
import re
import string
from html import unescape
import random
import tiktoken
import numpy as np
import httpx

# array of questions
questions = []

# load questions from file
print("Loading questions from questions.txt...")
with open(os.path.join(os.path.dirname(__file__), 'questions.txt')) as f:
    for line in f:
        questions.append(line.strip())
print("Loaded", len(questions), "questions...")


# shuffle entries
print("\nShuffle questions...")
random.shuffle(questions)

# create training set and validation set
print("Creating training and validation sets...")
n = len(questions)
train_entries = questions[:int(n*0.9)]
val_entries = questions[int(n*0.9):]

# tokenizer
enc = tiktoken.get_encoding("gpt2")

# loop through questions, tonize, add EOS token
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
