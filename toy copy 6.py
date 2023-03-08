from torch import nn
import torch

x = torch.rand(2, 2)
print(type(x))  # prints the truncated tensor
# print(x)
torch.set_printoptions(threshold=10_000)
print(type(x))
print(x)  # prints the whole tensor

print(torch.max(x))
print(torch.min(x))
print(torch.mean(x))
