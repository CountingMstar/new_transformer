import torch
import torch.nn as nn

# Create an nn.Linear module
linear = nn.Linear(10, 5)

# Access the weight and bias attributes
weight = linear.weight
bias = linear.bias

# Print the shape of the weight and bias tensors
print("Weight shape:", weight.shape)
print("Bias shape:", bias.shape)

print(weight)
print(bias)

a = weight.shape
print(a)
# b = weight.view(3, 10)
# print(b.shape)
# print(b)

# chunk
# c = torch.chunk(weight, 5, dim=1)
# print()
# # print(c.shape)
# print(type(c[0]))
# print(c[0].shape)
# print(c[0])

# split
s = torch.split(weight, 3, dim=1)
print()
# print(c.shape)
print(type(s[0]))
print(s[0].shape)
print(s[0])
