from unittest import TestCase

from layer import Layer
from my_mulitlayer_perceptron import MyMLP, InputFormatException
import numpy as np

np.random.seed(1)


INPUT_LAYER_SIZE = 2000
OUTPUT_LAYER = Layer(1000, 'sigmoid')
H1 = Layer(3000, 'relu')
H2 = Layer(2000, 'tanh')
LAYERS = [H1, H2, OUTPUT_LAYER]


class TestMyMLP(TestCase):
    def __init__(self, *args, **kwargs):
        super(TestMyMLP, self).__init__(*args, **kwargs)
        self.mlp = MyMLP(INPUT_LAYER_SIZE, LAYERS)

    def test_should_model_have_proper_structure(self):
        model = self.mlp.model
        self.assertEqual(model[0].weights.shape, (H1.num_nodes, INPUT_LAYER_SIZE))
        self.assertEqual(model[0].bias.shape, (H1.num_nodes, 1))
        self.assertEqual(model[1].weights.shape, (H2.num_nodes, H1.num_nodes))
        self.assertEqual(model[1].bias.shape, (H2.num_nodes, 1))
        self.assertEqual(model[2].weights.shape, (OUTPUT_LAYER.num_nodes, H2.num_nodes))
        self.assertEqual(model[2].bias.shape, (OUTPUT_LAYER.num_nodes, 1))

    def test_should_model_weights_have_uniform_dist(self):
        model = self.mlp.model
        for layer in model:
            self.assertAlmostEqual(np.average(layer.weights), 0.5, 1)
            self.assertAlmostEqual(np.average(layer.bias), 0.5, 1)

    def test_should_raise_input_size_exception(self):
        input_vec = np.ones((INPUT_LAYER_SIZE - 1, 1))
        with self.assertRaises(InputFormatException):
            self.mlp.feed_forward(input_vec)

    def test_should_not_rise_input_size_exception(self):
        input_vec = np.ones((INPUT_LAYER_SIZE, 1))
        self.mlp.feed_forward(input_vec)

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
        mlp.print_weights()

        result = mlp.feed_forward(input_vec)
        self.assertEqual(result[0][0], 4)
        self.assertEqual(result[1][0], 0)

