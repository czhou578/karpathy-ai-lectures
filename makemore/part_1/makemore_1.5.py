import torch
import torch.nn.functional as F

'''
E04: we saw that our 1-hot vectors merely select a row of W, so producing these vectors explicitly feels wasteful. 
Can you delete our use of F.one_hot in favor of simply indexing into rows of W?
'''

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


for w in words:
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
train_loss_arr = []
dev_loss_arr = []

for k in range(100):
    xsidx = xs[:, 0]
    xsidx2 = xs[:, 1]

    logits = W[xsidx][xsidx2]
    loss = F.cross_entropy(logits, ys)
    # counts = logits.exp()
    # probs = counts / counts.sum(1, keepdims=True)
    # loss = -probs[torch.arange(ytrain.shape[0]), ytrain].log().mean() + 0.01*(W**2).mean()
    
    if k >= 90:
        with torch.no_grad():
            xenc = F.one_hot(xdev, num_classes=27).float()
            logits = xenc.view(-1, 27) @ W
            dev_loss = F.cross_entropy(logits, ys)
            dev_loss_arr.append(dev_loss.item())

    train_loss_arr.append(loss.item())
    W.grad = None
    loss.backward()

    W.data += -0.2 * W.grad

print("Mean of the last 10 training loss: ", sum(train_loss_arr)/10)
print("Mean of the last 10 dev set loss: ", sum(dev_loss_arr)/10)  

test_loss_arr = []
with torch.no_grad():
    for j in range(10):
        xenc = F.one_hot(xtest, num_classes=27).float()
        logits = xenc.view(-1, 27) @ W
        loss = F.cross_entropy(logits, ys)
    test_loss_arr.append(loss.item())

print("Mean of the test set loss: ", sum(test_loss_arr)/10)  

