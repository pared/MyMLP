from unittest import TestCase
from my_mulitlayer_perceptron import MyMLP, InputFormatException
import numpy as np

np.random.seed(1)


INPUT_LAYER_SIZE = 2000
OUTPUT_LAYER_SIZE = 1000
H1 = 3000
H2 = 2000
HIDDEN_LAYERS = [H1, H2]


class TestMyMLP(TestCase):
    def __init__(self, *args, **kwargs):
        super(TestMyMLP, self).__init__(*args, **kwargs)
        self.mlp = MyMLP(INPUT_LAYER_SIZE, OUTPUT_LAYER_SIZE, HIDDEN_LAYERS)

    def test_should_model_have_proper_structure(self):
        model = self.mlp.model
        self.assertEqual(model[0].weights.shape, (H1, INPUT_LAYER_SIZE))
        self.assertEqual(model[0].bias.shape, (H1, 1))
        self.assertEqual(model[1].weights.shape, (H2, H1))
        self.assertEqual(model[1].bias.shape, (H2, 1))
        self.assertEqual(model[2].weights.shape, (OUTPUT_LAYER_SIZE, H2))
        self.assertEqual(model[2].bias.shape, (OUTPUT_LAYER_SIZE, 1))

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

