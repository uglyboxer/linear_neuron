# Author: Cole Howard
#
# Tests for Network Class of linear_neuron project

import unittest
from source.network import Network


class NetworkTests(unittest.TestCase):

    def test_mse(self):
        network = Network(2, 3, [20, 20, 20], [10, 10, 10], 1)
        assert network.mse() == None
