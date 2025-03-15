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
block_size = 8 # context size
n_embd = 32
head_size = 4
device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
    # print(ix.shape)
    x = torch.stack([data[i: i + block_size] for i in ix])
    # print(x.shape)
    y = torch.stack([data[i + 1: i + block_size + 1] for i in ix])

    return x, y

xb, yb = get_batch('train')

# Model Architecture

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('mask', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        keys = self.key(x)
        queries = self.query(x)
        values = self.value(x)

        # attention_scores = queries * keys^T
        attention_scores = torch.matmul(queries, keys.transpose(1, 2))

        # divide since otherwise softmax will explode. Single token will dominate.
        attention_scores = attention_scores / (C ** 0.5)
        attention_scores = attention_scores.masked_fill(self.mask[:T, :T] == 0, float('-inf'))
        attention_probs = F.softmax(attention_scores, dim=-1)
        output = attention_probs @ values

        return output

class MultiHeadAttention(nn.Module):
    def __init__(self, head_size, num_heads):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.project = nn.Linear(num_heads * head_size, n_embd)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        head_outputs = [head(x) for head in self.heads]
        output = torch.cat(head_outputs, dim=-1)
        output = self.project(output)
        output = self.dropout(output)
        return output

class ModelBase(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.transformer = Transformer()
        self.positional_embedding = nn.Embedding(block_size, n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, input, targets=None): # input is B * T (batch size * block size)

        token_embeddings = self.token_embedding(input)
        positional_embeddings = self.positional_embedding(torch.arange(input.shape[1], device=device))
        # print("positional and token embeddings", positional_embeddings.shape, token_embeddings.shape)
        embeddings = token_embeddings + positional_embeddings
        output = self.transformer(embeddings)

        logits = self.lm_head(output)

        if targets is None:
            return logits, None
        else:
            B, T, C = logits.shape

            logits = logits.view(B * T, C)
            targets = targets.view(B * T)

            loss = F.cross_entropy(logits, targets)

            return logits, loss

        # logits = self.token_embedding(input)

        # B, T, C = logits.shape
        # logits = logits.view(B * T, C)
        # targets = targets.view(B * T)
        # loss = F.cross_entropy(logits, targets)

        # return logits, loss
    
    def generate(self, idx, max_tokens):
        # Get rid of list append - work with tensors instead
        for _ in range(max_tokens):
            # crop idx to block_size tokens if longer
            idx_cond = idx[:, -block_size:]
            # get predictions
            logits, _ = self(idx_cond, None)
            # print("logits shape", logits.shape)
            # focus on last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # get probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append to sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx


class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(0.1)
        )
    
    def forward(self, x):
        return self.net(x)

class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.layerNorm1 = nn.LayerNorm(n_embd)
        self.layerNorm2 = nn.LayerNorm(n_embd)
        self.attention = MultiHeadAttention(head_size, 8)
        self.feed_forward = FeedForward()

    def forward(self, x):
        x = x + self.attention(self.layerNorm1(x))
        x = x + self.feed_forward(self.layerNorm2(x))

        return x

m = ModelBase(vocab_size)
m = m.to(device)
optimizer = torch.optim.Adam(m.parameters(), lr=0.001)
logits, loss = m(xb, yb)
# print("loss is", loss.item())

for i in range(1000):
    xb, yb = get_batch('train')
    logits, loss = m(xb, yb)
    # print("loss is", loss.item())
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if i % 100 == 0:
        print(f"step {i}: loss {loss.item()}")

print("the final loss is", loss.item())

# ...after training loop...

# Initialize context with a single token
context = torch.zeros((1, 1), dtype=torch.long, device=device)
# Generate 100 new tokens
generated_tokens = m.generate(context, max_tokens=100)[0].tolist()
# Convert tokens to text
generated_text = ''.join([itos[i] for i in generated_tokens])
print(generated_text)


