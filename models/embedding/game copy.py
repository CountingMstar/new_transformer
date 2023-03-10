import torch
from sklearn.metrics.pairwise import cosine_similarity

class PE_GAME:
    def __init__(self, d_model, max_len):
        super(PE_GAME, self).__init__()
        self.state = torch.zeros(max_len, d_model)
        self.state_size = int(d_model * max_len) 
        self.action_size = self.state_size
        self.max_len = max_len
        self.d_model = d_model

    def reset(self):
        self.state = torch.zeros(self.max_len, self.d_model)
        return self.state

    def step(self, action):
        action = action.view(self.max_len, self.d_model)
        state = self.state
        self.state = self.state + action
        # tensor안의 값을 -1 ~ 1로 제한
        self.state = torch.clamp(self.state, min=-1.0, max=1.0)
        next_state = self.state

        reward_list = []
        for i in range(self.max_len):
            m = round((self.max_len-1)/2)
            for j in range(self.max_len - (i+1)):
                k = j + i + 1
                """
                (i, k) == (행, 렬)
                """
                vector1 = action[i].tolist()
                vector2 = action[k].tolist()
                """
                중간값 m을 기준으로 이보다 작은면 +reward, 크면 -reward
                -> 거리가 가까우면 유사도를 크게, 멀면 유사도를 작게 하기위해
                """
                if k <= m+i:
                    similarity = cosine_similarity([vector1], [vector2])
                if k > m+i:
                    similarity = -cosine_similarity([vector1], [vector2])
                reward_list.append(similarity)
        reward = sum(reward_list)[0][0]
        # print(reward_list)
        done = False
        return next_state, reward, done

# d_model = 3
# max_len = 4
# action = torch.tensor([[-1.1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]])
# # action = torch.tensor([[1, 2, 3, 1, 2, 2, 7, 8, 9, 10, 11, 12]])

# game = PE_GAME(d_model, max_len)
# game.reset()
# step = game.step(action)
# print(step)
