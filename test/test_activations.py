from unittest import TestCase

from activations import ACTIVATIONS


class TestActivations(TestCase):
    def test_relu(self):
        relu = ACTIVATIONS['relu']
        self.assertEqual(relu.a(-100), 0)
        self.assertEqual(relu.a(100), 100)
        self.assertEqual(relu.d_a(-100), 0)
        self.assertEqual(relu.d_a(100), 1)