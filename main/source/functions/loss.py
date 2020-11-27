import math

from numpy.core.multiarray import ndarray
import numpy as np


class Loss:
    @staticmethod
    def categorical_cross_entropy(output: ndarray, label: int):
        return -np.log(output[label])

    @staticmethod
    def gradients_categorical_cross_entropy(label: int, size: int, output: ndarray) -> ndarray:
        gradients = np.zeros(size)
        gradients[label] = -1 / output[label]
        return gradients
