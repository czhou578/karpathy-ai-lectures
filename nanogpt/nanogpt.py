'''
train transformer on chunks of data / sample random chunks then train. chunks of max length (block size)
mini batches stacked up in single tensor (for gpu efficiency)

'''
import torch
import torch.nn as nn
from torch.nn import functional as F


torch.manual_seed(1337)
block_size = 8
batch_size = 4
vocab_size = 65
n_embed = 32

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    print(''.join(chars))
    print(vocab_size)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])
print(encode("hii there"))
print(decode(encode("hii there")))

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    data = train_data if split == "train" else val_data 
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])

    return x, y



xb, yb = get_batch('train')

class Bigram(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)
    
    def forward(self, idx, targets=None):
        tok_embed = self.token_embedding_table(idx)
        logits = self.lm_head(tok_embed) #B T C

        if targets == None:
            return
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, loss = self(idx)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        
        return idx

m = Bigram(vocab_size)
logits, loss = m(xb, yb)

optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)
for steps in range(100): # increase number of steps for good results...

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print(loss.item())

print(decode(m.generate(idx = torch.zeros((1, 1), dtype=torch.long), max_new_tokens=100)[0].tolist()))

# print(logits.shape)
# print(loss)

    