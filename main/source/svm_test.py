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


number = 100
x_features = np.random.randn(number)

y_labels = np.random.randn(number)

x_features = rbf(1, x_features, weights=np.ones(x_features.shape[0]))

plt.plot(x_features, y_labels, 'ro')
plt.ylabel('some numbers')
plt.show()
