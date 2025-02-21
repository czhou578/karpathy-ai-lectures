import torch
import torch.nn.functional as F

words = open("names.txt", 'r').read().splitlines()

N = torch.zeros((27, 27, 27), dtype=torch.int32)

chars = sorted(list(set(''.join(words))))
stoi = {s: i + 1 for i, s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s, i in stoi.items()}
g = torch.Generator().manual_seed(2147483647)

W = torch.rand((27, 27), generator=g, requires_grad=True)

xs, ys = [], []

words_len = len(words)
train_idx = int(0.80 * words_len)
dev_idx = int(0.90 * words_len)

for w in train:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2, ch3 in zip(chs, chs[1:], chs[2:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        ix3 = stoi[ch3]
        xs.append([ix1, ix2])
        ys.append(ix3)

xs = torch.tensor(xs)
ys = torch.tensor(ys)

xtrain, ytrain = xs[:train_idx], ys[:train_idx]
xdev, ydev = xs[train_idx:dev_idx], ys[train_idx:dev_idx]
xtest, ytest = xs[dev_idx:], ys[dev_idx:]


num = xs.nelement()

for k in range(50):
    xenc = F.one_hot(xs, num_classes=27).float()
    logits = xenc.view(-1, 27) @ W
    counts = logits.exp()
    probs = counts / counts.sum(1, keepdims=True)
    print(ys.shape[0])
    loss = -probs[torch.arange(ys.shape[0]), ys].log().mean() + 0.01*(W**2).mean()
    
    W.grad = None
    loss.backward()

    W.data += -20 * W.grad

    print(loss.item())

