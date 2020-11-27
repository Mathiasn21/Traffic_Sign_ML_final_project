import unittest

import mnist

from ml_models.cnn.layer.conv2d import Conv2D


class TestConv2D(unittest.TestCase):
    training_images = mnist.test_images()[:10]
    training_labels = mnist.test_labels()[:10]
    num_labels = 10

    def test_conv2d(self):
        conv2d = Conv2D(8, (3, 3))
        shape = self.training_images.shape[1:]
        print(shape)
        print(conv2d.forward_propagate((self.training_images[0] / 255) - 0.5))
        print(conv2d.forward_propagate(self.training_images[0]).shape)


if __name__ == '__main__':
    unittest.main()
