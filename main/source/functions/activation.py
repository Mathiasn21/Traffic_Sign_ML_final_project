from math import exp


class Activation:
    @staticmethod
    def softmax(z: list):
        z_length = len(z)
        sigma_list = [0.] * z_length
        for i in range(0, z_length):
            a = exp(z[i])
            b = 0
            for j in range(0, z_length):
                b += exp(z[j])

            if b == 0:
                raise Exception('Division by null exception.')

            sigma_list[i] = a / b
        return sigma_list
