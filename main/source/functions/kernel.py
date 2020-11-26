import math
from scipy.spatial import distance
import numpy as np


class Kernel:

    @staticmethod
    def linear_kernel(x, xi):
        return sum(x * xi)

    @staticmethod
    def polynomial_kernel(x, xi, c, degree=2):
        return (sum(x * xi) + c) ** degree


    @staticmethod
    def rbf_kernel(data, x, xi):
        sigma_squared = np.var(data)
        gamma = 1 / (2 * sigma_squared)
        dst = distance.euclidean(x, xi)
        rbf = math.exp(-gamma * dst)
        return rbf
