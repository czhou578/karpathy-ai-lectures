import torch
import torch.nn.functional as F
import random


words = open('names.txt', 'r').read().splitlines()
words[:8]
chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}
vocab_size = len(itos)