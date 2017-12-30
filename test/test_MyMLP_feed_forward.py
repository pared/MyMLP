from unittest import TestCase

from my_mulitlayer_perceptron import MyMLP
from layer import Layer
import numpy as np

class TestMyMLPFeedForward(TestCase):
    def test_feed_forward(self):
        input_vec = np.ones((2, 1))
        mlp = MyMLP(2, [Layer(2, 'relu')])

        #override weights
        new_weights = np.zeros(shape=(2, 2))
        new_weights[0] = [2, 1]
        new_weights[1] = [-1, -2]
        mlp.model[0].weights = new_weights

        new_bias = np.zeros(shape=(2, 1))
        new_bias[0] = [1]
        new_bias[1] = [-1]

        mlp.model[0].bias = new_bias

        output, layers_inputs = mlp.feed_forward(input_vec)

        np.testing.assert_array_equal(output, np.array([[4],
                                                        [0]]))

        self.assertEqual(list(layers_inputs.keys()), [2])
        np.testing.assert_array_equal(layers_inputs[2], np.array([[4],
                                                                  [-4]]))
