# from torch import nn
# import torch

# x = torch.rand(2, 2)
# print(type(x))  # prints the truncated tensor
# # print(x)
# torch.set_printoptions(threshold=10_000)
# print(type(x))
# print(x)  # prints the whole tensor

# print(torch.max(x))
# print(torch.min(x))
# print(torch.mean(x))

import pickle
import torch

with open('qkv/q.pickle', 'rb') as f:
    q = pickle.load(f)
with open('qkv/k_t.pickle', 'rb') as f:
    k_t = pickle.load(f)
with open('qkv/root_d_tensor.pickle', 'rb') as f:
    root_d_tensor = pickle.load(f)
with open('qkv/score.pickle', 'rb') as f:
    score = pickle.load(f)
with open('qkv/qk.pickle', 'rb') as f:
    qk = pickle.load(f)

print(q.shape)
print(k_t.shape)
print(qk.shape)
print(root_d_tensor)
print(score.shape)

print((q@k_t).shape)
print((q@k_t)[0][0][0])
print(qk.shape)
print(qk[0][0][0])

s = (q @ k_t) / root_d_tensor
print(s.shape)
print(s[0][0][0])
print(score[0][0][0])
print(score[0][0][0]*8)


# qk = q @ k_t
# print(qk.shape)

# a = torch.tensor([[1, 2, 3]])
# print(a.shape)
# print(a)
# b = a.transpose(0, 1)
# print(b.shape)
# print(b)

# print(b @ a)
# print((b @ a)/10)

