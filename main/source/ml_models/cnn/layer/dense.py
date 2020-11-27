from typing import List, Type
import numpy as np
from numpy import ndarray
from numpy.core._multiarray_umath import ndarray
from numpy.core.multiarray import ndarray

from functions.activation import Activation
from ml_models.cnn.layer.layer import Layer


class Dense(Layer):
    previous_totals: ndarray
    previous_data_shape: tuple
    activation: str
    nodes: int
    weights: ndarray
    biases: ndarray
    previous_data: ndarray
    learning_rate: float = .005

    def __init__(self, data_length: int, nodes: int, activation: str = "relu"):
        self.activation = activation
        self.weights = np.random.randn(data_length, nodes) / data_length
        self.biases = np.zeros(nodes)

    def forward_propagate(self, data_in: ndarray) -> ndarray:
        self.previous_data_shape = data_in.shape
        data_in = data_in.flatten()
        self.previous_data = data_in

        data_length, nodes = self.weights.shape

        # Calculates the dot product of two list like objects and add bias
        totals = np.dot(data_in, self.weights) + self.biases
        output = []
        self.previous_totals = totals

        if self.activation == 'softmax':
            output = Activation.softmax(totals)
        elif self.activation == 'relu':
            output = Activation.softmax(totals)
        return output

    def back_propagate(self, gradients: ndarray) -> ndarray:
        """
        pd = partial derivative.
        :param gradients:
        :return: ndarray gradients
        """

        for i, gradient in enumerate(gradients):
            if gradient == 0:
                continue

            pd_out_pd_t = Activation.softmax_partial_derivative(self.previous_totals, i)

            pd_t_pd_w = self.previous_data
            pd_t_pd_b = 1
            pd_t_pd_inputs = self.weights

            # Gradients of loss with respect to totals
            pd_l_pd_t = gradient * pd_out_pd_t

            # Gradients of loss with respect to weights/biases/inputs
            pd_l_pd_w = pd_t_pd_w[np.newaxis].T @ pd_l_pd_t[np.newaxis]
            pd_l_pd_b = pd_l_pd_t * pd_t_pd_b
            pd_l_pd_inputs = pd_t_pd_inputs @ pd_l_pd_t

            # Update weights / biases
            self.weights -= self.learning_rate * pd_l_pd_w
            self.biases -= self.learning_rate * pd_l_pd_b

            # Reshape due to matrix operations
            return pd_l_pd_inputs.reshape(self.previous_data_shape)
