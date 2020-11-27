from ml_models.cnn.layer.layer import Layer


class Conv2D(Layer):
    filters: int
    kernel_size: tuple
    activation: str

    def __init__(self, filters: int, kernel_size: tuple):
        self.kernel_size = kernel_size
        self.filters = filters
