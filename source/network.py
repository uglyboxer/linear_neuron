# Author: Cole Howard
# 
# network.py is a basic implementation of a one layer neural network, to
# examine an implementation of backpropagation.  The intent is to make it
# extensible as a tool for exploring neural nets under more general
# circumstances.

class Network:

    def __init__(self, neuron_count, vector_size, epoch_nums, learn_iter=1):
        self.neuron_count = neuron_count
        self.vector_size = vector_size
        self.epoch_nums = epoch_nums
        self.learn_iter = learn_iter