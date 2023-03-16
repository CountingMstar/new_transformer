import pickle
import torch

time = 1000
layer_num = 5
version = str((time, layer_num))
# print(input_x.shape)
# print(output_x.shape)
# print(ressum_x.shape)
# print(norm_x.shape)

with open("input_x" + str(version) + ".pkl", 'rb') as f:
    input_x = pickle.load(f)
with open("output_x" + str(version) + ".pkl", "rb") as f:
    output_x = pickle.load(f)
with open("ressum_x" + str(version) + ".pkl", "rb") as f:
    ressum_x = pickle.load(f)
with open("norm_x" + str(version) + ".pkl", "rb") as f:
    norm_x = pickle.load(f)

def info(x):
    # print(x)
    mean = torch.mean(x)
    std = torch.std(x)
    min = torch.min(x)
    max = torch.max(x)
    print(mean)
    print(std)
    print(min)
    print(max)

print('input_x')
info(input_x)

print('output_x')
info(output_x)

print('ressum_x')
info(ressum_x)

print('norm_x')
info(norm_x)
