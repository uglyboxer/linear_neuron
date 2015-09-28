# Author: Cole Howard
#
# Tests for Network Class of linear_neuron project

import unittest
from source.network import Network


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

    def setup(self):
        self.neuron = Neuron(3)

    def test_append_bias(self):
        pass

    def test_dot_product(self):
        pass

    def test_sigmoid(self):
        pass

    def test_update_weights(self):
        pass

    def test_apply_backprop(self):
        pass

    def test_send_backprop(self):
        pass

    def test_fires(self):
        pass
