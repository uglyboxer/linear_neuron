# Author: Cole Howard
#
# network.py is a basic implementation of a one layer neural network, to
# examine an implementation of backpropagation.  The intent is to make it
# extensible as a tool for exploring neural nets under more general
# circumstances.


class Network:

    def __init__(self, neuron_count, vector_size, train, test_set,
                 epoch_nums, layers=1, learn_iter=1):
        """ A Network instance will create layers of neurons for the implementa-
        tion of neural network.

        Args:
            neuron_count: int
            vector_size: int (training or test)
            train: a list or numpy array
            test_set: a list or numpy array
            epoch_nums: an int (the number of times backprop should occur)
            layers: int
            learn_iter: an int (the number of times before backprop)
        """

        self.neuron_count = neuron_count    # Per layer
        self.vector_size = vector_size
        self.train = train
        self.test_set = test_set
        self.epoch_nums = epoch_nums
        self.layers = layers
        self.learn_iter = learn_iter

    def mse(self):     # Calculate Mean Squared Errorpass
        pass

    def diff_log_func(self):   # Differentiated Log Funciton
        pass

