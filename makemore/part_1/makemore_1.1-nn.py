import torch
import torch.nn.functional as F

words = open("names.txt", 'r').read().splitlines()

N = torch.zeros((27, 27, 27), dtype=torch.int32)

chars = sorted(list(set(''.join(words))))
stoi = {s: i + 1 for i, s in enumerate(chars)}
stoTri = {}
stoi['.'] = 0
itos = {i:s for s, i in stoi.items()}

chars.insert(0, '.')

i = 0
for ch1 in chars:
    for ch2 in chars:
        stoTri[ch1 + ch2] = i
        i += 1

# print(f"Length of stoTri keys is, {len(stoTri.keys())}")
# print(stoTri)

xs, ys = [], []

for w in words:
    chs = ['.'] + ['.'] + list(w) + ['.']
    # print("chs is, ", chs)

    for ch1, ch2, ch3 in zip(chs, chs[1:], chs[2:]):
        ix1 = stoTri[ch1 + ch2]
        ix2 = stoi[ch3]

        # print(ch1, ch2, ch3)

        xs.append(ix1)
        ys.append(ix2)
    

xs = torch.tensor(xs)
# print(f'xs shape is {xs.shape} and xs is {xs}')
ys = torch.tensor(ys)
num = xs.nelement()

# print(f"Output labels ys is, {ys}")

W = torch.rand((729, 27), requires_grad=True)


for k in range(100):
    xenc = F.one_hot(xs, num_classes=729).float()
    logits = xenc @ W
    counts = logits.exp()
    probs = counts / counts.sum(1, keepdim=True)
    # print(ys.shape[0])
    loss = -probs[torch.arange(num), ys].log().mean() + 0.01*(W**2).mean()
    
    W.grad = None
    loss.backward()

    W.data += -50 * W.grad

    print(loss.item())

# xenc = F.one_hot(xs, num_classes=729).float()

# logits = xenc @ W
# counts = logits.exp()
# probs = counts / counts.sum(1, keepdim=True)
# print("Probs shape is, ", probs.shape)

# loss = -probs[torch.arange(5), ys].log().mean()
# print(loss.item())

'''
Lessons:
key: concatenate strings together using special map and use it to index.
Errors encountered: RuntimeError: Class values must be smaller than num_classes

'''









