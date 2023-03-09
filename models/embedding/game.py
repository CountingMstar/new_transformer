import torch
from sklearn.metrics.pairwise import cosine_similarity

class PE_GAME:
    def __init__(self, d_model, max_len):
        super(PE_GAME, self).__init__()
        self.state = torch.zeros(max_len, d_model)
        self.state_size = int(d_model * max_len) 
        self.action_size = self.state_size

    def reset(self):
        self.state = torch.zeros(max_len, d_model)
        print(self.state)

    def step(self, action):
        action = action.view(max_len, d_model)
        state = self.state
        self.state = self.state + action
        next_state = self.state

        print('action')
        print(action)
        print(action[0])
        vector1 = action[0].tolist()
        vector2 = action[1].tolist()
        print(vector1)
        similarity = cosine_similarity([vector1], [vector2])
        # reward = torch.tensor(similarity)
        reward = similarity

        done = False
        return next_state, reward, done

d_model = 3
max_len = 4
action = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]])

game = PE_GAME(d_model, max_len)
game.reset()
step = game.step(action)
print(step)
