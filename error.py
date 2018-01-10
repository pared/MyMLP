import numpy as np


class Error:
    def __init__(self, function, gradient):
        self.error_function = function
        self.error_function_gradient = gradient

    def error(self, expected, actual):
        return self.error_function(expected, actual)

    def gradient(self, expected, actual):
        return self.error_function_gradient(expected, actual)


def mse():
    return Error(lambda expected, actual: 1 / 2 * sum(((expected - actual) ** 2))[0],
                 lambda expected, actual: actual - expected)
