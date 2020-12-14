import time
from typing import Union
import numpy as np
from numpy.core.multiarray import ndarray
from ml_models.cnn.layer.layer import Layer


class Conv2D(Layer):
    last_input: ndarray
    filters: ndarray
    generated_filters: bool = False
    learning_rate: float = .01
    num_filters: int
    kernel_size: tuple
    padding: int
    strides: int
    activation: str

    def __init__(self,
                 num_filters: int,
                 kernel_size: tuple,
                 padding: int = 0, strides: int = 1,
                 activation: str = 'relu'):
        self.activation = activation
        self.kernel_size = kernel_size
        self.padding = padding
        self.strides = strides
        self.num_filters = num_filters
        self.weight_init = True

    def forward_propagate(self, data_in: ndarray) -> ndarray:
        """
        Performs a forward propagation of current conv layer
        using input: data_in. Which must be of 2D ndarray.
        Returns 3D ndarray given as: (height, width, filters)
        :param data_in:
        """
        if not self.generated_filters:
            self.generate_filters()

        self.last_input = data_in
        kernel_h = self.kernel_size[0]
        kernel_w = self.kernel_size[1]
        data_h, data_w = data_in.shape

        output = np.zeros((data_h - (kernel_h - 1), data_w - (kernel_w - 1), self.num_filters))

        for region, i, j in self.generate_regions(data_in, self.kernel_size):
            for k, kernel in enumerate(self.filters):
                output[i, j, k] = np.sum(np.multiply(region, kernel))
        return output

    def back_propagate(self, gradients: ndarray) -> ndarray:
        kernel_w = self.kernel_size[0]
        kernel_h = self.kernel_size[1]
        gradients_shape: tuple = gradients.shape

        # Generates filter gradients given previous layer gradients
        filter_gradients = np.zeros((self.num_filters, kernel_h, kernel_w))
        for region, i, j in self.generate_regions(self.last_input, gradients_shape):
            for k, kernel in enumerate(self.filters):
                filter_gradients[k, i, j] = np.sum(np.multiply(region, gradients[:, :, k]))

        # Generates feature gradients given next layer gradients and rotated filters
        rotated_filters = np.rot90(self.filters, 2, axes=(1, 2))
        input_shape: tuple = self.last_input.shape
        padding_d = (kernel_w - 1)
        padding_w = gradients_shape[0] + padding_d * 2
        padding_h = gradients_shape[1] + padding_d * 2

        feature_gradients = np.zeros((input_shape[0], input_shape[1], self.num_filters))
        altered_d_shape = np.zeros((padding_w, padding_h, self.num_filters))
        altered_d_shape[padding_d:-padding_d, padding_d:-padding_d, :] = gradients

        for region, i, j in self.generate_regions(altered_d_shape, self.kernel_size):
            for k, kernel in enumerate(rotated_filters):
                feature_gradients[i, j, k] = np.sum(np.multiply(region[:, :, k], kernel))

        self.filters -= filter_gradients * self.learning_rate
        return feature_gradients

    def generate_regions(self, data_input: ndarray, region_shape: tuple) -> ndarray:
        """
        Generator function.\n
        Generates all possible regions from input using kernel size
        Using valid padding by default.
        :rtype: ndarray: consisting of regions
        :param data_input: data will be divided into regions
        :param region_shape: shape describing each region
        """
        shape = data_input.shape
        data_h = shape[0]
        data_w = shape[1]
        region_h = region_shape[0]
        region_w = region_shape[1]

        for i in range(0, data_h - (region_h - 1), self.strides):
            for j in range(0, data_w - (region_w - 1), self.strides):
                region = data_input[i:(i + region_h), j:(j + region_w)]
                yield region, i, j

    def generate_filters(self, input_depth: Union[None, int] = None):
        h = self.kernel_size[0]
        w = self.kernel_size[1]

        # creates a 4D tensor input given data shape
        if input_depth is None:
            self.filters = np.random.randn(self.num_filters, h, w) / (h * w)
        else:
            self.filters = np.random.randn(self.num_filters, h, w, input_depth) / (h * w)

        self.generated_filters = True
