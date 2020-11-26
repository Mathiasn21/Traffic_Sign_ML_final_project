from typing import List
import numpy as np
from numpy import ndarray

from functions.activation import Activation


class Dense:
    nodes: int
    weights: ndarray[float]
    biases: ndarray[float]
    activation: str

    def __init__(self, data_length: int, nodes: int, activation: str = "relu"):
        self.activation = activation
        self.weights = np.random.randn(data_length, nodes) / data_length
        self.biases = np.zeros(nodes)

    def forward_propagate(self, data_in: ndarray) -> ndarray:
        data_in = data_in.flatten()
        data_length, nodes = self.weights.shape

        # Calculates the dot product of two list like objects and adds bias
        totals = np.dot(data_in, self.weights) + self.biases
        output = []

        if self.activation == 'softmax':
            output = Activation.softmax(totals)
        elif self.activation == 'relu':
            output = Activation.softmax(totals)
        return output

    def back_propagate(self):
        print("")
