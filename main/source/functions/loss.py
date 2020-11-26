import math


class Loss:
    @staticmethod
    def categorical_cross_entropy(y: list, y_hat: list):
        y_length = len(y)
        y_hat_length = len(y_hat)
        if y_length != y_hat_length:
            raise Exception('Mismatching lengths of lists.')

        loss = 0
        for i in range(0, y_length):
            loss += y[i] * math.log(y_hat[i])
        return -loss
