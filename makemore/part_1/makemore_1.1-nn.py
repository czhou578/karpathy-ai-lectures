import torch
import torch.nn.functional as F

words = open("names.txt", 'r').read().splitlines()

N = torch.zeros((27, 27, 27), dtype=torch.int32)

chars = sorted(list(set(''.join(words))))
stoi = {s: i + 1 for i, s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s, i in stoi.items()}

xs, ys = [], []

for w in words[:1]:
    chs = ['.'] + list(w) + ['.']
    print(chs)

    for ch1, ch2, ch3 in zip(chs, chs[1:], chs[2:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        ix3 = stoi[ch3]

        print(ch1, ch2, ch3)

        xs.append(ch1)
        ys.append(ch2)
        zs.append(ch3)
    

xs = torch.tensor(xs)
ys = torch.tensor(ys)
zs = torch.tensor(zs)

W = torch.rand((54, 27), requires_grad=True)








