import torch
import torch.nn as nn
import torch.nn.functional as F

"""
DONE - Create encode/decode functions to convert between text and tokens
Prepare your training data by tokenizing the text
Split your data into training and validation sets
Create data loading functions that generate batches of sequences
Define your model architecture:
Create a token embedding layer
Add positional encodings
Implement self-attention blocks
Add feed-forward networks
Add layer normalization
Initialize the model and define loss function (typically cross-entropy)
Set up training loop with optimizer (usually AdamW)
Implement evaluation and metrics tracking
Add text generation functionality to sample from the model
Implement model checkpointing to save/load progress
"""

batch_size = 12
block_size = 8

with open('jfk-speeches.txt', encoding="utf-8") as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = { ch:i for i,ch in enumerate(chars) } # char to int
itos = { i:ch for i,ch in enumerate(chars) } # int to char

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: [itos[i] for i in l]

data = torch.tensor(encode(text), dtype=torch.long)

n = int(len(data) * 0.8)
training_data = data[:n]
validation_data = data[n:]

def get_batch(split):
    training_data if split == 'train' else validation_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i: i + block_size] for i in ix])
    y = torch.stack([data[i + 1: i + block_size + 1] for i in ix])

    return x, y

xb, yb = get_batch('train')

# Model Architecture

class ModelBase(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, vocab_size)

    def forward(self, input, targets): # input is B * T (batch size * block size)
        logits = self.token_embedding(input)

        B, T, C = logits.shape
        logits = logits.view(B * T, C)
        targets = targets.view(B * T)
        loss = F.cross_entropy(logits, targets)

        return logits, loss

m = ModelBase(vocab_size)
logits, loss = m(xb, yb)
print("loss is", loss.item())




