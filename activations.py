from scipy.special import expit
import numpy as np


class Activation():
    def __init__(self, name, activation, derivative):
        self.name = name
        self.a = activation
        self.d_a = derivative


# TODO maybe remove duplicate call of  expit in derivative?
SIGMOID = Activation('sigmoid', lambda x: expit(x), lambda x: expit(x) * (1 - expit(x)))
RELU = Activation('relu', lambda x : np.maximum(0, x),lambda x:  (x > 0))
TANH = Activation('tanh', lambda x: np.tanh(x), lambda x: 1 - np.tanh(x)**2)

ACTIVATIONS = {a.name: a for a in [SIGMOID, RELU, TANH]}