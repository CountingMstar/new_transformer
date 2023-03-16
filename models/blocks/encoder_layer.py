"""
@author : Hyunwoong
@when : 2019-10-24
@homepage : https://github.com/gusdnd852
"""
from torch import nn

from models.layers.layer_norm import LayerNorm
from models.layers.multi_head_attention import MultiHeadAttention
from models.layers.position_wise_feed_forward import PositionwiseFeedForward

import pickle


class EncoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, n_head, drop_prob):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.norm1 = LayerNorm(d_model=d_model)
        self.dropout1 = nn.Dropout(p=drop_prob)

        self.ffn = PositionwiseFeedForward(
            d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob
        )
        self.norm2 = LayerNorm(d_model=d_model)
        self.dropout2 = nn.Dropout(p=drop_prob)
        self.time = 0

    def forward(self, x, s_mask, layer_num):
        # 1. compute self attention
        _x = x
        # print(self.time)
        # print(_x[0][0].shape)
        # print(layer_num)
        input_x = _x[0][10]
        version = str((self.time, layer_num))

        x = self.attention(q=x, k=x, v=x, mask=s_mask)

        output_x = x[0][10]

        # 2. add and norm
        # print(" ")
        # print(x[0][0].shape)

        ressum_x = (x + _x)[0][10]
        # print(_x + x)
        x = self.dropout1(x)

        # x = self.norm1(x + _x)
        x = self.norm1(x)
        norm_x = self.norm1(ressum_x)

        # print(input_x.shape)
        # print(output_x.shape)
        # print(ressum_x.shape)
        # print(norm_x.shape)

        if self.time % 100 == 0:
            with open("result_resnet/input_x" + str(version) + ".pkl", "wb") as f:
                pickle.dump(input_x, f)
            with open("result_resnet/output_x" + str(version) + ".pkl", "wb") as f:
                pickle.dump(output_x, f)
            with open("result_resnet/ressum_x" + str(version) + ".pkl", "wb") as f:
                pickle.dump(ressum_x, f)
            with open("result_resnet/norm_x" + str(version) + ".pkl", "wb") as f:
                pickle.dump(norm_x, f)

        # 3. positionwise feed forward network
        _x = x
        x = self.ffn(x)

        # 4. add and norm
        x = self.dropout2(x)
        # x = self.norm2(x + _x)
        x = self.norm2(x)

        self.time += 1
        return x
