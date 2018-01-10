import numpy as np
from typing import List, Tuple

from activations import ACTIVATIONS
from exceptions import InputFormatException
from layer import Layer, ModelLayer
from utils import pretty_str_float, pretty_str_list


class MyMLP:
    def __init__(self, input_size: int, layers: List[Layer]):
        self.input_size = input_size
        self.layers = layers
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

    def print_weights(self):
        for layer in self.model:
            for i in range(layer.weights.shape[0]):
                print("W" + str(i) + ": " + pretty_str_list(layer.weights[i], 3) + " b" + str(i) + ": " + str(pretty_str_float(layer.bias[i], 3)))
            print("-----------------------")

    def summary(self):
        print("IN: %i" % self.input_size)
        for index, layer in enumerate(self.layers[:-1]):
            print("H%i: %i" % (index + 1, layer.num_nodes))
        print("OUT: %i" % self.layers[-1].num_nodes)

    def feed_forward(self, input_vec):
        '''
        :param input_vec: input vector for neural network
        :return: tuple, first element is result of propagation of input through NN,
        second is a map storing z's: {layer_number: input_of_layer_in_current_pass}
        I assume that input is first layer, so map lowest key is 2 (there is no need to store input)
        '''

        self.__check_input_size(input_vec.shape)
        layers_inputs = []

        result = input_vec
        for index,model_layer in enumerate(self.model):
            result = np.matmul(model_layer.weights, result)
            result += model_layer.bias

            #save layer input
            layers_inputs.append(result)

            result = model_layer.activation.a(result)
        return result, layers_inputs

    def __initialize_model_with_random_weights(self):
        model = []
        for index, layer in enumerate(self.layers):
            width = self.__get_layer_nodes_number(index - 1)
            height = layer.num_nodes

            bias = np.random.rand(height, 1)
            W = np.random.rand(height, width)

            model.append(ModelLayer(W, bias, ACTIVATIONS[layer.activation]))

        return model

    def __get_layer_nodes_number(self, index):
        if index > len(self.layers) - 1:
            raise Exception('Trying to get number of nodes from non-existing layer')
        if index is -1:
            return self.input_size
        else:
            return self.layers[index].num_nodes

    def __check_input_size(self, input_shape):
        if input_shape != (self.input_size, 1):
            raise InputFormatException("Wrong input size, expected: %s , actual: %s" % (self.input_size, input_shape))


