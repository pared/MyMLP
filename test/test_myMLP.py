from unittest import TestCase
from my_mulitlayer_perceptron import MyMLP
import numpy as np

np.random.seed(1)


INPUT_LAYER_SIZE = 2
OUTPUT_LAYER_SIZE = 1
H1 = 3
H2 = 5
HIDDEN_LAYERS = [H1, H2]


class TestMyMLP(TestCase):
    def __init__(self, *args, **kwargs):
        super(TestMyMLP, self).__init__(*args, **kwargs)
        self.mlp = MyMLP(INPUT_LAYER_SIZE, OUTPUT_LAYER_SIZE, HIDDEN_LAYERS)

    def test_model_structure(self):
        model = self.mlp.model
        self.assertEqual(model[0].weights.shape, (H1, INPUT_LAYER_SIZE))
        self.assertEqual(model[0].bias.shape, (H1, 1))
        self.assertEqual(model[1].weights.shape, (H2, H1))
        self.assertEqual(model[1].bias.shape, (H2, 1))
        self.assertEqual(model[2].weights.shape, (OUTPUT_LAYER_SIZE, H2))
        self.assertEqual(model[2].bias.shape, (OUTPUT_LAYER_SIZE, 1))




