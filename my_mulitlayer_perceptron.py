import numpy as np
from typing import List, Tuple

from utils import pretty_str_float, pretty_str_list


class MyMLP:
    def __init__(self, input_size: int, output_size: int, layers: List[int]):
        '''
        :param input_size: number of input parameters
        :param output_size: number of output parameters
        :param layers: consequent elements indicate number of nodes in hidden layers
         [3 , 2] means two hidden, dense layers with 3 and 2 nodes respectively
        '''
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layers = layers
        self.model = self.__initialize_model_with_random_weights()

    def fit(self, X: List[List], Y: List[List], epochs: int):
        '''
        Sample for xor
        :param X: [[0, 0]
                   [1, 1]
                   [1, 0]
                   [0, 1]]
        :param Y: [[0]
                   [0]
                   [1]
                   [1]]
        :param epochs: 10000
        :return:
        '''
        for epoch in range(epochs):
            pass

    def summary(self):
        for layer in self.model:
            for i in range(layer.weights.shape[0]):
                print("W" + str(i) + ": " + pretty_str_list(layer.weights[i], 3) + " b" + str(i) + ": " + str(pretty_str_float(layer.bias[i], 3)))
            print("-----------------------")

    def __initialize_model_with_random_weights(self):
        model = []
        layers = self.hidden_layers + [self.output_size]
        for index, nodes_number in enumerate(layers):
            width = self.__get_layer_nodes_number(index - 1)
            height = nodes_number

            bias = np.random.rand(height, 1)
            W = np.random.rand(height, width)

            model.append(Layer(W, bias))

        return model


    def __get_layer_nodes_number(self, index):
        if index > len(self.hidden_layers) - 1:
            raise Exception('Trying to get number of nodes from non-existing layer')
        if index is -1:
            return self.input_size
        else:
            return self.hidden_layers[index]


class Layer:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias