from numpy.core.multiarray import ndarray
import numpy as np


class CNN:
    epochs: int
    learning_rate: float
    training_features: ndarray
    training_labels: ndarray

    def __init__(self,
                 epochs: int,
                 test_data: ndarray,
                 training_features: ndarray,
                 training_labels: ndarray,
                 learning_rate: float = .005):
        self.training_features = training_features
        self.training_labels = training_labels
        self.test_data = test_data
        self.epochs = epochs
        self.learning_rate = learning_rate

    def fit(self):
        print("")
        # TODO: implement method

    def test(self, test_features: ndarray, test_labels: ndarray):
        print("")
        # TODO: implement method

