import matplotlib.pyplot as plt
import pickle

with open('scores.pkl', 'rb') as f:
    total_list = pickle.load(f)

with open('avg_scores.pkl', 'rb') as f:
    total_avg_list = pickle.load(f)

# plt.plot(total_list)
# plt.plot(total_avg_list)

plt.plot(total_list, "b", label="validation")
plt.plot(total_avg_list, "r", label="train")
plt.legend(loc="lower right")

plt.xlabel("epoch")
plt.ylabel("score")
plt.title("result")
plt.grid(True, which="both", axis="both")
plt.savefig("result")

plt.show()
