import math


class Distribution:
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + math.exp(-x))
