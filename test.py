import torch

a = torch.ones([10, 10])
b = torch.rand([10, 10])

print(a[:, None, :].size())

