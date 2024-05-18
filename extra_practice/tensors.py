import torch

'''
torch.stack
torch.view
torch.cat
torch.arange
torch.sum

'''

B,T,C = 4,8,2 # batch, time, channels
x = torch.randn(B,T,C)
print('x is ', x)

xbow = torch.zeros((B,T,C))


for b in range(B):
    for t in range(T):
        xprev = x[b,:t+1] # (t,C)
        print('xprev is ', xprev)
        xbow[b,t] = torch.mean(xprev, 0)
        print('average is, ', + xbow[b, t])
        # break
    break


