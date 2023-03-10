import matplotlib.pyplot as plt
import numpy as np
import torch

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
print('######pos_encoding#####')
print(pos_encoding.shape)
print(pos_encoding)

plt.figure(figsize=(10, 8))
plt.pcolormesh(pos_encoding, cmap='RdBu')
plt.xlabel('Embedding Dimension')
plt.xlim((0, d_model))
plt.ylabel('Sequence Position')
plt.ylim((0, max_len))
plt.colorbar()
plt.title('Positional Encoding Visualization')
plt.show()



