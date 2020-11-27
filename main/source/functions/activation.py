from math import exp

import numpy as np
from numpy.core._multiarray_umath import ndarray


class Activation:
    @staticmethod
    def softmax(totals: ndarray) -> ndarray:
        # Core math function for softmax activation
        exp_array = np.exp(totals)
        return exp_array / np.sum(exp_array, axis=0)

    @staticmethod
    def softmax_partial_derivative(total_c: float, total_k: float, totals: list, c: int) -> float:
        s: ndarray = np.sum(totals)
        exp_total_c: float = exp(total_c)

        gradients = -exp_total_c * exp(total_k) * (s ** -2)
        gradients[c] = (exp_total_c * (s - exp_total_c)) * (s ** -2)
        return gradients
