import matplotlib.pyplot as plt
import numpy as np
import torch
from game import PE_GAME
import pickle

def positional_encoding(d_model, max_len):
    encoding = torch.zeros(max_len, d_model)
    pos = torch.arange(0, max_len)
    pos = pos.float().unsqueeze(dim=1)

    _2i = torch.arange(0, d_model, step=2).float()

    encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
    encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))

    return encoding[:, :] 

d_model = 10
max_len = 10

pos_encoding = positional_encoding(d_model, max_len)
pos_encoding = pos_encoding.view(1, -1)
zero_encoding = torch.zeros(d_model*max_len)
one_encoding = torch.ones(d_model*max_len)

# proposed method
score = -0.1
with open('state_'+ str(score) +'.pkl', 'rb') as f:
    total_list = pickle.load(f)
# print('#####MAX#####')
# print(total_list)

# linear
a = 2/9
x = [-1 for i in range(10)]
x = [x[i]+a*(i) for i in range(10)]
x = [[i]*10 for i in x]
linear = torch.tensor(x)

env = PE_GAME(d_model, max_len)
env.reset()
# next_state, reward, done = env.step(pos_encoding)
# next_state, reward, done = env.step(zero_encoding)
# next_state, reward, done = env.step(one_encoding)
# next_state, reward, done = env.step(linear)
next_state, reward, done = env.step(total_list)

print("#####reward#####")
print(reward)

print('######pos_encoding#####')
pos_encoding = next_state.view(d_model, max_len)
print(next_state.shape)
print(next_state)

plt.figure(figsize=(10, 8))
plt.pcolormesh(pos_encoding, cmap='RdBu')
plt.xlabel('Embedding Dimension')
plt.xlim((0, d_model))
plt.ylabel('Sequence Position')
plt.ylim((0, max_len))
plt.colorbar()
plt.title('Positional Encoding Visualization')
plt.savefig("result_heatmap_total_list2", path="/home/moonstar/python/NLP/new_transformer/models/embedding/result2")
# plt.show()



