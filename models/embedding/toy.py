# import torch
# import numpy as np

# from sklearn.metrics.pairwise import cosine_similarity
# import numpy as np

# # Create two sample vectors
# vector1 = [1, 2, 3]
# vector2 = [3, 2, 1]

# # Calculate cosine similarity between the two vectors
# print(vector1)
# similarity = cosine_similarity([vector1], [vector2])

# print(similarity)

# seed = 0
# torch.manual_seed(seed)

# print(torch.manual_seed(seed))
# print(np.random.seed(seed))



# a = torch.zeros(1, 9)
# a = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]])
# print(a)
# b = a.view(4, 3)
# print(b)

####################################
# import torch

# # create a tensor
# x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])

# # clip the elements of the tensor between -1 and 1
# x_clipped = torch.clamp(x, min=-1.0, max=1.0)

# # print the clipped tensor
# print(x_clipped)

########################################
# import numpy as np
# import torch

# experiences = [np.array([[-0.84212846, -0.53927696,  2.5129163 ]]), np.array([[-0.84212846, -0.53927696,  2.5129163 ]])]
# print(experiences)
# states = torch.from_numpy(np.vstack([e for e in experiences if e is not None])).float()
# print(states)

# experiences = [torch.tensor([[0.6964, 0.7565, 0.8348, 0.9327, 0.7931, 0.9525, 0.8462, 0.8121, 0.8072, 0.9547, 0.8246, 0.8835]]), 
#                 torch.tensor([[0.6964, 0.7565, 0.8348, 0.9327, 0.7931, 0.9525, 0.8462, 0.8121, 0.8072, 0.9547, 0.8246, 0.8835]])]
# print(experiences)
# states = torch.from_numpy(np.vstack([e for e in experiences if e is not None])).float()
# print(states)

#######################

# r = [5, 2, 3, 4, 6, 2, 1, 100, 0, 50, 30]
r = [-5, -2, -3, -4, -6, -2, -1, -100, -0, -50, -30]
print(r)
s = [i for i in range(len(r))]
print(s)
best_states = []

for i in range(len(r)):
    reward = r[i]
    state = s[i]
    state_plus_reward = [reward, state]
    print(state_plus_reward)

    if len(best_states) < 5:
        best_states.append(state_plus_reward)
    elif len(best_states) == 5:
        re = [rs[0] for rs in best_states]
        min_index = re.index(min(re))
        if min(re) < reward:
            del best_states[min_index]
            best_states.append(state_plus_reward)

    print('========')
    print(best_states)

print("######best_states")
print(best_states)
re = [rs[0] for rs in best_states]
print(re)

bs = [(rs[0], rs[1]) for rs in best_states]
print(bs)
for r, s in bs:
    print('=========')
    print(r)
    print(s)