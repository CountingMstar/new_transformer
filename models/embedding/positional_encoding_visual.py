import matplotlib.pyplot as plt
import numpy as np

def positional_encoding(seq_length, embedding_size):
    pos = np.arange(seq_length)[:, np.newaxis]
    i = np.arange(embedding_size)[np.newaxis, :]
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(embedding_size))
    angle_rads = pos * angle_rates
    sin = np.sin(angle_rads[:, 0::2])
    cos = np.cos(angle_rads[:, 1::2])
    pos_encoding = np.concatenate([sin, cos], axis=-1)
    return pos_encoding

seq_length = 50
embedding_size = 10

pos_encoding = positional_encoding(seq_length, embedding_size)

plt.figure(figsize=(10, 8))
plt.pcolormesh(pos_encoding, cmap='RdBu')
plt.xlabel('Embedding Dimension')
plt.xlim((0, embedding_size))
plt.ylabel('Sequence Position')
plt.ylim((0, seq_length))
plt.colorbar()
plt.title('Positional Encoding Visualization')
plt.show()



