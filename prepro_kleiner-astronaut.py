"""
Downloads and tokenizes the kleiner-astronaut dataset.
- The download is from HuggingFace datasets.
- The tokenization is GPT-2 tokenizer with tiktoken

The output is written to a newly created data/ folder.
The script prints:

Tokenizing val split...
Saved 19043638 tokens to data/kleiner-astronaut_val.bin
Tokenizing train split...
Saved 925653391 tokens to data/kleiner-astronaut_train.bin

And runs in 1-2 minutes two depending on your internet
connection and computer. The .bin files are raw byte
streams of int32 numbers indicating the token ids.
"""

import os
import glob
import json
import random
import requests
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from datasets import load_dataset
import tiktoken
import numpy as np

DATA_CACHE_DIR = "data"
enc = tiktoken.get_encoding("gpt2")
encode = lambda s: enc.encode_ordinary(s)

def process_dataset(dataset, ds_index):
    subset=dataset[ds_index]
    eot = enc._special_tokens['<|endoftext|>'] # end of text token
    shuffled_subset = subset.shuffle(seed=42)
    all_tokens = []
    for sample in shuffled_subset:
        text = sample["text"]
        text = text.strip()  # get rid of leading/trailing whitespace
        #print(text)
        tokens = encode(text)
        all_tokens.append(eot)
        all_tokens.extend(tokens)
    return all_tokens

def tokenize():
    dataset = load_dataset("jotschi/kleiner-astronaut")
    print(dataset)
    for set in ["train", "test"]:
        all_tokens = []
        all_tokens.extend(process_dataset(dataset, set))
        all_tokens_np = np.array(all_tokens, dtype=np.int32)
        with open("data/kleiner-astronaut_" + set + ".bin", "wb") as f:
                f.write(all_tokens_np.tobytes())

if __name__ == "__main__":
    tokenize()

    # Prints:
    # Tokenizing val split...
    # Saved 19043638 tokens to data/TinyStories_val.bin
    # Tokenizing train split...
    # Saved 925653391 tokens to data/TinyStories_train.bin
