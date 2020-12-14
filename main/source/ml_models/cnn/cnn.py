import time

import numpy as np
from numpy.core.multiarray import ndarray

from functions.loss import Loss
from ml_models.cnn.layer.layer import Layer


class CNN:
    layers: list
    epochs: int
    learning_rate: float
    training_features: ndarray
    training_labels: ndarray

    def __init__(self,
                 epochs: int,
                 training_features: ndarray,
                 training_labels: ndarray,
                 learning_rate: float = .05):

        self.training_features = training_features
        self.training_labels = training_labels
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.layers = list()

    def add(self, layer: Layer):
        self.layers.append(layer)

    def forward(self, image: ndarray, label):
        layer: Layer
        out: ndarray = (image / 255) - 0.5

        for i, layer in enumerate(self.layers):
            if not layer.weight_init:
                layer.init_weights(self.calc_data_length(out.shape))

            out = layer.forward_propagate(out)

        loss = Loss.categorical_cross_entropy(out, label)
        acc = 1 if np.argmax(out) == label else 0
        return out, loss, acc

    def back_propagation(self, initial_gradients: ndarray):
        gradients = initial_gradients

        for i in range(len(self.layers) - 1, -1, -1):
            layer: Layer = self.layers[i]
            gradients = layer.back_propagate(gradients)

    def fit(self):
        for epoch in range(0, self.epochs):
            print('--- Epoch: %d ---' % epoch)
            tot_loss = 0
            num_correct = 0

            # Initialize adams optimizer values
            beta_1 = 0.1
            beta_2 = 0.001
            epsilon = 1.E-9
            step_size = 0.0002
            v = None
            m = None

            t = 0
            w = 0.00000001

            for i, (feature, label) in enumerate(zip(self.training_features, self.training_labels)):
                if i % 100 == 99:
                    print(
                        '[Step %d] Past 100 steps: Average Loss %.3f | Accuracy: %d%%' %
                        (i + 1, tot_loss / 100, num_correct)
                    )
                    tot_loss = 0
                    num_correct = 0

                out, loss, acc = self.forward(feature, label)
                if v is None:
                    v = np.zeros(out.shape)
                    m = np.zeros(out.shape)

                tot_loss += loss
                num_correct += acc
                # Calculate initial gradient
                gradients = np.zeros(out.shape)
                gradients[label] = -1 / out[label]

                for gradient in gradients:
                    m = beta_1 * m + (1 - beta_1) * gradient
                    v = beta_2 * v + (1 - beta_2) * np.power(gradient, 2)
                    m_hat = m / (1 - np.power(beta_1, t))
                    v_hat = v / (1 - np.power(beta_2, t))
                    w = w - step_size * m_hat / (np.sqrt(v_hat) + epsilon)
                t += 1
                self.back_propagation(gradients)

    def test(self, test_features: ndarray, test_labels: ndarray):
        print("")
        # TODO: implement method

    @staticmethod
    def calc_data_length(shape: tuple):
        data_length = 1
        for item in shape:
            data_length *= item
        return data_length
