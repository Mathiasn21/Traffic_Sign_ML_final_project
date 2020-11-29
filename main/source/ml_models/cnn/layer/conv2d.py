from typing import Union

from numpy.core._multiarray_umath import ndarray

from ml_models.cnn.layer.layer import Layer
import numpy as np


class Conv2D(Layer):
    last_input: ndarray
    filters: ndarray
    generated_filters: bool = False
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
        for region, i, j in self.generate_regions(data_in):
            output[i, j] = np.sum(region * self.filters, axis=(1, 2))

        return output

    def back_propagate(self, gradients: ndarray) -> ndarray:
        print()

    def convolve(self, image, filt) -> ndarray:
        # TODO: Convolve image with filter
        # Eg: out(i, j) = SUM(filter(x, y) * img(x - i, y - j), i, j)
        kernel_h = self.kernel_size[0]
        kernel_w = self.kernel_size[1]
        image_h, image_w = image.shape

        feature_map = np.zeros((image_h - (kernel_h - 1), image_w - (kernel_w - 1), self.num_filters))


    def generate_regions(self, data_input: ndarray) -> ndarray:
        """
        Generates all possible regions from input using kernel size
        Using valid padding by default.
        :param data_input:
        """
        shape = data_input.shape
        data_h = shape[0]
        data_w = shape[1]
        kernel_h = self.kernel_size[0]
        kernel_w = self.kernel_size[1]

        for i in range(0, data_h - (kernel_h - 1), self.strides):
            for j in range(0, data_w - (kernel_w - 1), self.strides):
                region = data_input[i:(i + kernel_h), j:(j + kernel_w)]
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
