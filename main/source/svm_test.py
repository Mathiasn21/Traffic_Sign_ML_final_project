from matplotlib import pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from numpy.core.multiarray import ndarray
from sklearn import datasets
from sklearn.datasets import make_blobs, make_circles


# https://github.com/fbeilstein/simplest_smo_ever/blob/main/simple_svm.ipynb
class SVM:
    def __init__(self, kernel='linear', C=10000.0, max_iter=100000, degree=3, gamma=1):
        self.kernel = {'poly': lambda x, y: np.dot(x, y.T) ** degree,
                       'rbf': lambda x, y: np.exp(-gamma * np.sum((y - x[:, np.newaxis]) ** 2, axis=-1)),
                       'linear': lambda x, y: np.dot(x, y.T)}[kernel]
        self.C = C
        self.max_iter = max_iter

    def restrict_to_square(self, t, v0, u):
        t = (np.clip(v0 + t * u, 0, self.C) - v0)[1] / u[1]
        return (np.clip(v0 + t * u, 0, self.C) - v0)[0] / u[0]

    def fit(self, X, y):
        self.X = X.copy()
        self.y = y * 2 - 1  # shifts the label center from 0,1 to -1,1
        self.lambdas = np.zeros_like(self.y, dtype=float)
        self.K = self.kernel(self.X, self.X) * self.y[:, np.newaxis] * self.y

        for _ in range(self.max_iter):
            for idxM in range(len(self.lambdas)):
                idxL = np.random.randint(0, len(self.lambdas))

                Q = self.K[[[idxM, idxM], [idxL, idxL]], [[idxM, idxL], [idxM, idxL]]]
                v0 = self.lambdas[[idxM, idxL]]
                k0 = 1 - np.sum(self.lambdas * self.K[[idxM, idxL]], axis=1)
                u = np.array([-self.y[idxL], self.y[idxM]])
                t_max = np.dot(k0, u) / (np.dot(np.dot(Q, u), u) + 1E-15)
                self.lambdas[[idxM, idxL]] = v0 + u * self.restrict_to_square(t_max, v0, u)

        idx, = np.nonzero(self.lambdas > 1E-15)
        self.b = np.sum((1.0 - np.sum(self.K[idx] * self.lambdas, axis=1)) * self.y[idx]) / len(idx)

    def decision_function(self, X):
        return np.sum(self.kernel(X, self.X) * self.y * self.lambdas, axis=1) + self.b


def rbf(gamma, feature_vectors: ndarray, weights: ndarray):
    shape = feature_vectors.shape
    classifiers = np.zeros(shape[0])

    for i, vector_i in enumerate(feature_vectors):
        s = 0
        for vector_j, weight in zip(feature_vectors, weights):
            s += np.exp(-gamma * np.linalg.norm(vector_i - vector_j)) * weight
        classifiers[i] = s
    return classifiers


X, y = make_blobs(n_samples=150, centers=2, cluster_std=1.4)
X, y = make_circles(150, factor=.1, noise=.1)


def test_plot(X, y, svm_model, axes, title):
    plt.axes(axes)
    xlim = [np.min(X[:, 0]), np.max(X[:, 0])]
    ylim = [np.min(X[:, 1]), np.max(X[:, 1])]
    xx, yy = np.meshgrid(np.linspace(*xlim, num=700), np.linspace(*ylim, num=700))
    rgb = np.array([[210, 0, 0], [0, 0, 150]]) / 255.0

    svm_model.fit(X, y)
    z_model = svm_model.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
    plt.contour(xx, yy, z_model, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])
    plt.contourf(xx, yy, np.sign(z_model.reshape(xx.shape)), alpha=0.3, levels=2, cmap=ListedColormap(rgb), zorder=1)
    plt.title(title)
    plt.show()


fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))
test_plot(X, y, SVM(kernel='rbf', C=10, max_iter=60, degree=3, gamma=1), axs[0], 'OUR ALGORITHM')
