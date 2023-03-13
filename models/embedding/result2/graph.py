import matplotlib.pyplot as plt
import pickle

d_model = 4
max_len = 50
version = (d_model, max_len, 50)

# d_model = 5
# max_len = 50
# version = (d_model, max_len)

with open('scores'+ str(version) +'.pkl', 'rb') as f:
    total_list = pickle.load(f)

with open('avg_scores'+ str(version) +'.pkl', 'rb') as f:
    total_avg_list = pickle.load(f)

# plt.plot(total_list)
# plt.plot(total_avg_list)

plt.plot(total_list, "b", label="score")
plt.plot(total_avg_list, "r", label="avg_score")
plt.legend(loc="lower right")

plt.xlabel("epoch")
plt.ylabel("score")
plt.title("result")
plt.grid(True, which="both", axis="both")
# plt.xlim((0, 2000))
plt.savefig("result")

plt.show()

# score = 24.800000021457606
# with open('state_reward-'+ str(score) +'.pkl', 'rb') as f:
#     total_list = pickle.load(f)

# print(total_list)
