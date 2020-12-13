import numpy as np
from numpy.core.multiarray import ndarray


class Activation:
    @staticmethod
    def softmax(totals: ndarray) -> ndarray:
        # Core math function for softmax activation
        exp_array = np.exp(totals)
        return exp_array / np.sum(exp_array, axis=0)

    @staticmethod
    def softmax_partial_derivative(totals: ndarray, c: int) -> ndarray:
        exp_totals: ndarray = np.exp(totals)
        s: ndarray = np.sum(exp_totals)

        gradients = -exp_totals[c] * exp_totals * (s ** -2)
        gradients[c] = (exp_totals[c] * (s - exp_totals[c])) * (s ** -2)
        return gradients
