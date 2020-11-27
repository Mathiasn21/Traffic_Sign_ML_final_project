from numpy.core.multiarray import ndarray


class Layer(object):
    weight_init = False
    def init_weights(self, data_length: int):
        """
        Initializes weights with random values
        :param data_length:
        """
        pass

    def forward_propagate(self, data_in: ndarray) -> ndarray:
        """
        Executes a single forward propagation of the layer.
        Requires input from previous layer and outputs output depending on the layer.
        :param data_in:
        """
        pass

    def back_propagate(self, gradients: ndarray) -> ndarray:
        """
        Executes a back propagation using gradients as input and returns computed gradients
        :param gradients:
        """
        pass
