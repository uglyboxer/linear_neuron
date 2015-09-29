# Author: Cole Howard
#
# Tests for Network Class of linear_neuron project

import unittest
from ..source.network import append_bias, Network, Neuron
from ..source.network import Neuron 


class NetworkHelperTests(unittest.TestCase):

    def test_append_bias(self):
        self.assertEqual(append_bias([0, 0, 0]), [0, 0, 0, 1])

class NetworkTests(unittest.TestCase):

    def setUp(self):
        self.network = Network(2, 3, [20, 20, 20], [10, 10, 10], 1)

    def test_mse(self):
        assert self.network.mse([1, 1], [.75, .5]) == .15625
        assert self.network.mse([0], [0]) == 0

    def test_diff_log_function(self):
        pass

    def test_learn_run(self):
        pass

    def test_run_unseen(self):
        pass

    def test_report_results(self):
        pass


class NeuronTests(unittest.TestCase):

    def setUp(self):
        self.neuron = Neuron(3)

    def test_dot_product(self):
        assert self.neuron._dot_product([4, 3, 2], [2, 3, 4]) == 25

    def test_sigmoid(self):
        assert self.neuron._sigmoid(0) == .5
        assert self.neuron._sigmoid(1000000) == 1
        self.assertAlmostEqual(self.neuron._sigmoid(-705), 0)

    def test_update_weights(self):
        pass

    def test_apply_backprop(self):
        pass

    def test_send_backprop(self):
        pass

    def test_fires(self):
        pass



