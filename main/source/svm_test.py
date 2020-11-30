import random

import mnist
from matplotlib import pyplot as plt
import numpy as np
from numpy.core.multiarray import ndarray


def rbf(gamma, feature_vectors: ndarray, weights: ndarray):
    shape = feature_vectors.shape
    classifiers = np.zeros(shape[0])

    for i, vector_i in enumerate(feature_vectors):
        s = 0
        for vector_j, weight in zip(feature_vectors, weights):
            s += np.exp(-gamma * np.linalg.norm(vector_i - vector_j)) * weight
        classifiers[i] = s
    return classifiers


num = 100
#  = np.random.randn(number, 1)

x_features = mnist.test_images()[:100]
y_labels = mnist.test_labels()[:100]

x_features = rbf(1, x_features, weights=np.random.randn(x_features.shape[0]))

plt.plot(y_labels, x_features, 'ro')
plt.ylabel('some numbers')
plt.show()
