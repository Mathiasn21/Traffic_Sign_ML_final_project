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
                 learning_rate: float = .005):
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

            for i, (feature, label) in enumerate(zip(self.training_features, self.training_labels)):
                if i % 100 == 99:
                    print(
                        '[Step %d] Past 100 steps: Average Loss %.3f | Accuracy: %d%%' %
                        (i + 1, tot_loss / 100, num_correct)
                    )
                    tot_loss = 0
                    num_correct = 0

                out, loss, acc = self.forward(feature, label)

                tot_loss += loss
                num_correct += acc
                # Calculate initial gradient
                gradients = np.zeros(out.shape)
                gradients[label] = -1 / out[label]


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
