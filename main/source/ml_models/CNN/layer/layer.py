from numpy.core.multiarray import ndarray


class Layer(object):
    def forward_propagate(self, data_in: ndarray) -> ndarray:
        """
        Executes a single forward propagation of the layer.
        Requires input from previous layer and outputs output depending on the layer.
        :param data_in:
        """
        pass

    def back_propagate(self, gradients: float) -> ndarray:
        """
        Executes a back propagation using gradients as input and returns computed gradients
        :param gradients:
        """
        pass
