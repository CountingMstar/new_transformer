import torch
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Create two sample vectors
vector1 = [1, 2, 3]
vector2 = [3, 2, 1]

# Calculate cosine similarity between the two vectors
print(vector1)
similarity = cosine_similarity([vector1], [vector2])

print(similarity)

seed = 0
torch.manual_seed(seed)

print(torch.manual_seed(seed))
print(np.random.seed(seed))



a = torch.zeros(1, 9)
a = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]])
print(a)
b = a.view(4, 3)
print(b)