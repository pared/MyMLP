from unittest import TestCase

from error import mse
import numpy as np


class ErrorTest(TestCase):
    def test_mse(self):
        error_fun = mse()
        expected = np.array([[0],
                             [1]])
        actual = np.array([[1],
                           [0]])
        self.assertEqual(error_fun.error(expected, actual), 1)
        np.testing.assert_array_equal(error_fun.gradient(expected, actual), np.array([[1],
                                                                         [-1]]))