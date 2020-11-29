import unittest

import mnist
from numpy.core.multiarray import ndarray

from ml_models.cnn.layer.conv2d import Conv2D
import numpy as np


class TestConv2D(unittest.TestCase):
    training_images = mnist.test_images()[:10]
    training_labels = mnist.test_labels()[:10]
    num_labels = 10

    def test_conv2d(self):
        conv2d = Conv2D(8, (3, 3))
        shape: tuple = self.training_images.shape[1:]
        img = self.training_images[0]
        print(shape)
        h = 3
        w = 3
        kernel_size = 2

        output = np.zeros((shape[0] - (3 - 1), shape[1] - (3 - 1), 2))
        filters = np.random.randn(2, 3, 3) / (np.prod(shape))

        for region, i, j in conv2d.generate_regions(img):
            for k, kernel in enumerate(filters):
                output[i, j, k] = np.sum(np.multiply(region, kernel))

        output2 = np.zeros((shape[0] - (3 - 1), shape[1] - (3 - 1), 2))
        for region, i, j in conv2d.generate_regions(img):
            output2[i, j] = np.sum(region * filters, axis=(1, 2))

        ########################################################################
        # Generates filter gradients given previous layer gradients
        output3 = np.zeros((kernel_size, h, w))
        for region, i, j in self.generate_regions(img):
            for k, kernel in enumerate(filters):
                output3[k, i, j] = np.sum(np.multiply(region, output2[:, :, k]))
        ########################################################################
        # print(conv2d.forward_propagate((self.training_images[0] / 255) - 0.5))
        # print(conv2d.forward_propagate(self.training_images[0]).shape)
        print(output.shape)
        print(output2.shape)
        print(output3.shape)

        print(filters)

        ########################################################################
        # Generates feature gradients given previous layer gradients and rotated filter
        rotated_filters = np.rot90(filters, 2, axes=(1, 2))
        output4 = np.zeros((shape[0], shape[1], kernel_size))
        altered_d_shape = np.zeros((26 + (3 - 1)*2, 26 + (3 - 1)*2, 2))
        altered_d_shape[2:-2, 2:-2, :] = output2

        for region, i, j in conv2d.generate_regions(altered_d_shape):
            for k, kernel in enumerate(rotated_filters):
                output4[i, j, k] = np.sum(np.multiply(region[:, :, k], kernel))
        ########################################################################
        print(output4.shape)

        for row in output3:
            for col in row:
                print(col, end="\t")
            print()

    def generate_regions(self, data_input: ndarray) -> ndarray:
        """
        Generates all possible regions from input using kernel size
        Using valid padding by default.
        :param data_input:
        """
        shape = data_input.shape
        kernel_h = 26
        kernel_w = 26

        for i in range(0, shape[0] - (kernel_h - 1), 1):
            for j in range(0, shape[1] - (kernel_w - 1), 1):
                region = data_input[i:(i + kernel_h), j:(j + kernel_w)]
                yield region, i, j


if __name__ == '__main__':
    unittest.main()
