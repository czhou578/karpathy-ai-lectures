import torch
import torch.nn.functional as F

words = open("names.txt", 'r').read().splitlines()

N = torch.zeros((27, 27, 27), dtype=torch.int32)

chars = sorted(list(set(''.join(words))))
stoi = {s: i + 1 for i, s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s, i in stoi.items()}

for w in words:
    chs = ['.'] + list(w) + ['.']

    for ch1, ch2, ch3 in zip(chs, chs[1:], chs[2:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        ix3 = stoi[ch3]

        N[ix1, ix2, ix3] += 1

P = (N + 1).float()
P /= P.sum(2, keepdim=True)

log_likelihood = 0.0
n = 0

for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2, ch3 in zip(chs, chs[1:], chs[2:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        ix3 = stoi[ch3]

        prob = P[ix1, ix2, ix3]
        logprob = torch.log(prob)
        log_likelihood += logprob
        n += 1

nll = -log_likelihood
print('nll is, ', nll / n) #avg log likelihood

'''
Answer: Loss is 2.0927
'''

# g = torch.Generator().manual_seed(2147483647)

# for i in range(5):
  
#   out = []
#   ix = 0
#   while True:
    
#     # ----------
#     # BEFORE:
#     #p = P[ix]
#     # ----------
#     # NOW:
#     xenc = F.one_hot(torch.tensor([ix]), num_classes=27).float()
#     logits = xenc @ W # predict log-counts
#     counts = logits.exp() # counts, equivalent to N
#     p = counts / counts.sum(1, keepdims=True) # probabilities for next character
#     # ----------
    
#     ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
#     out.append(itos[ix])
#     if ix == 0:
#       break
#   print(''.join(out))