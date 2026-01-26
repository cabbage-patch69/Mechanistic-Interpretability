import torch


x = torch.randint(4,3,5)
vals, inds = torch.topk(x, dim=0) 
print(inds.shape)