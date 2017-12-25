from unittest import TestCase

from activations import ACTIVATIONS


class TestActivations(TestCase):
    def test_relu(self):
        relu = ACTIVATIONS['relu']
        self.assertEqual(relu.activation(-100), 0)
        self.assertEqual(relu.activation(100), 100)
        self.assertEqual(relu.derivative(-100), 0)
        self.assertEqual(relu.derivative(100), 1)