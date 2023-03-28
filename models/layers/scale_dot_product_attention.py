"""
@author : Hyunwoong
@when : 2019-10-22
@homepage : https://github.com/gusdnd852
"""
import math

from torch import nn

import pickle
import time



class ScaleDotProductAttention(nn.Module):
    """
    compute scale dot product attention

    Query : given sentence that we focused on (decoder)
    Key : every sentence to check relationship with Qeury(encoder)
    Value : every sentence same with Key (encoder)
    """

    def __init__(self):
        super(ScaleDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None, e=1e-12):
        # input is 4 dimension tensor
        # [batch_size, head, length, d_tensor]
        batch_size, head, length, d_tensor = k.size()

        # 1. dot product Query with Key^T to compute similarity
        k_t = k.transpose(2, 3)  # transpose

        print('@@@@@@@@@@@@@')
        print('q')
        print(q.shape)
        print('k_t')
        print(k_t.shape)
        print('qk')
        print((q @ k_t).shape)

        score = (q @ k_t) / math.sqrt(d_tensor)  # scaled dot product

        print('score')
        print(score.shape)


        with open('qkv/q.pickle', 'wb') as f:
            pickle.dump(q, f)
        with open('qkv/k_t.pickle', 'wb') as f:
            pickle.dump(k_t, f)
        with open('qkv/root_d_tensor.pickle', 'wb') as f:
            pickle.dump(math.sqrt(d_tensor), f)
        with open('qkv/score.pickle', 'wb') as f:
            pickle.dump(score, f)
        with open('qkv/qk.pickle', 'wb') as f:
            pickle.dump(q @ k_t, f)

        print('######time#####')
        time.sleep(1)

        # 2. apply masking (opt)
        if mask is not None:
            score = score.masked_fill(mask == 0, -10000)

        # 3. pass them softmax to make [0, 1] range
        score = self.softmax(score)

        # 4. multiply with Value
        v = score @ v

        return v, score
