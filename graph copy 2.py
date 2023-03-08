"""
@author : Hyunwoong
@when : 2019-12-18
@homepage : https://github.com/gusdnd852
"""

import matplotlib.pyplot as plt
import re


def read(name):
    f = open(name, "r")
    file = f.read()
    file = re.sub("\\[", "", file)
    file = re.sub("\\]", "", file)
    f.close()

    return [float(i) for idx, i in enumerate(file.split(","))]


def draw(mode):
    k = 50
    # k_str = str(k) + "-originallr"
    k_str = str(k)
    if mode == "loss":
        train = read("result/train_loss-" + k_str + ".txt")
        test = read("result/test_loss-" + k_str + ".txt")
        plt.plot(train, "r", label="train")
        plt.plot(test, "b", label="validation")
        plt.legend(loc="lower left")

    elif mode == "bleu":
        bleu = read("result/bleu-" + k_str + ".txt")
        plt.plot(bleu, "b", label="bleu score")
        plt.legend(loc="lower right")

    plt.xlabel("epoch")
    plt.ylabel(mode)
    plt.title("training result")
    plt.grid(True, which="both", axis="both")
    plt.savefig("saved/transformer-base/%s" % mode)
    plt.show()


if __name__ == "__main__":
    draw(mode="loss")
    draw(mode="bleu")
