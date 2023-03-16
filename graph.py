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
ver = [512]

for version in ver:
    type = "_noRESNET"
    version = str(version) + type
    print("%%%%%%%%%%%%")
    print(version)

    def draw(mode, version):
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
            plt.plot(bleu, "b", label="bleu score")
            plt.legend(loc="lower right")

        plt.xlabel("epoch")
        plt.ylabel(mode)
        plt.title("training result")
        plt.grid(True, which="both", axis="both")
        plt.savefig("saved/transformer-base/" + mode + "-" + version)
        # plt.show()
        plt.cla()

    if __name__ == "__main__":
        draw(mode="loss", version=version)
        draw(mode="bleu", version=version)
