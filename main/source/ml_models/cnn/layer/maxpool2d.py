import numpy as np
from numpy.core.multiarray import ndarray

from ml_models.cnn.layer.layer import Layer


class MaxPool2d(Layer):
    strides: int
    kernel_size: tuple
    last_input: ndarray

    def __init__(self, kernel_size: tuple = (2, 2), strides: int = 1):
        self.kernel_size = kernel_size
        self.strides = strides

    def forward_propagate(self, data_in: ndarray) -> ndarray:
        self.last_input = data_in
        h, w, num_filters = data_in.shape
        kernel_h, kernel_w = self.kernel_size
        output = np.zeros((h // kernel_h, w // kernel_w, num_filters))

        for im_region, i, j in self.generate_regions(data_in, self.kernel_size):
            output[i, j] = np.amax(im_region, axis=(0, 1))

        return output

    def back_propagate(self, gradients: ndarray) -> ndarray:
        """
        Performs a backward pass of the maxpool layer.
        Returns the loss gradient for this layer's inputs.
        - d_L_d_out is the loss gradient for this layer's outputs.
        """
        d_L_d_input = np.zeros(self.last_input.shape)

        for im_region, i, j in self.generate_regions(self.last_input, self.kernel_size):
            h, w, f = im_region.shape
            amax = np.amax(im_region, axis=(0, 1))

            for i2 in range(h):
                for j2 in range(w):
                    for f2 in range(f):
                        # If this pixel was the max value, copy the gradient to it.
                        if im_region[i2, j2, f2] == amax[f2]:
                            d_L_d_input[i * 2 + i2, j * 2 + j2, f2] = gradients[i, j, f2]

        return d_L_d_input

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
        region_h = region_shape[0]
        region_w = region_shape[1]
        data_h = shape[0] // region_h
        data_w = shape[1] // region_w

        for i in range(0, data_h - (region_h - 1), self.strides):
            for j in range(0, data_w - (region_w - 1), self.strides):
                region = data_input[(i * region_h):(i * region_h + region_h), (j * region_w):(j * region_w + region_w)]
                yield region, i, j
