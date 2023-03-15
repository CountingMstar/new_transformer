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


# ver = [10, 20, 30, 40]
# ver = [50, 100, 150, 200, 250, 300, 350, 400, 450]
# ver = [2, 4, 6, 8, 12, 14, 16, 18, 22, 24, 26, 28]
ver = [2, 4, 10, 20, 30, 40, 50]


def draw(mode):
    for version in ver:
        version = str(version)
        if mode == "loss":
            # train = read('result/train_loss.txt')
            # test = read('result/test_loss.txt')
            train = read("result/train_loss-" + version + ".txt")
            test = read("result/test_loss-" + version + ".txt")
            plt.plot(train, "r", label="train")
            plt.plot(test, "b", label="validation")
            plt.legend(loc="lower left")

        elif mode == "bleu":
            # bleu = read("result/bleu.txt")
            bleu = read("result/bleu-" + version + ".txt")
            # plt.plot(bleu, "b", label="bleu score")
            plt.plot(bleu, label="bleu score" + version)
            plt.legend(loc="lower left")
            # plt.xlim([0, 5])      # X축의 범위: [xmin, xmax]
            plt.ylim([35, 45])  # Y축의 범위: [ymin, ymax]

    plt.xlabel("epoch")
    plt.ylabel(mode)
    plt.title("training result")
    plt.grid(True, which="both", axis="both")
    plt.savefig("saved/transformer-base/total_" + mode + "-" + version)
    plt.show()


if __name__ == "__main__":
    # draw(mode="loss", version=version)
    draw(mode="bleu")
