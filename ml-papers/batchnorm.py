import torch
import torch.nn.functional as F
import random

words = open('names.txt', 'r').read().splitlines()

chars = sorted(list(set(''.join(words))))
stoi = {s: i + 1 for i, s in enumerate(chars)}
stoi['.'] = 0
itos = {i: s for i, s in stoi.items()}
vocab_size = len(itos)

