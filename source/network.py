# Author: Cole Howard
#
# network.py is a basic implementation of a one layer neural network, to
# examine an implementation of backpropagation.  The intent is to make it
# extensible as a tool for exploring neural nets under more general
# circumstances.

from math import e


class Network:

    def __init__(self, neuron_count, vector_size, train_set, test_set,
                 epoch_nums, layers=1, learn_iter=1):
        """ A Network instance will create layers of neurons for the implementa-
        tion of neural network.

        Args:
            neuron_count: int
            vector_size: int (training or test)
            train_set: a list or numpy array
            test_set: a list or numpy array
            epoch_nums: an int (the number of times backprop should occur)
            layers: int
            learn_iter: an int (the number of times before backprop)
        """

        self.neuron_count = neuron_count    # Per layer
        self.vector_size = vector_size
        self.train_set = train_set
        self.test_set = test_set
        self.epoch_nums = epoch_nums
        self.layers = layers
        self.learn_iter = learn_iter        # Size of leaning sets/iteration

    def mse(self, answer_vector, result_vector):
        """ Calculates the mean squared error between two vector_size

        Args:
            answer_vector: a list
            result_vector: a list

        Returns:
            a float
        """

        return sum([(answer_vector[i]-result_vector[i])**2 for i in
                    range(len(answer_vector))]) / len(answer_vector)

    def diff_log_func(self):   # Differentiated Log Funciton
        pass

    def learn_run(self):
        """ Runs an iteration through the neuron sets and adjust the weights
        appropriately.
        """ 
        for i in range(0, len(self.train_set), learn_iter):
#### FIX    
        

    def run_unseen(self):
        pass

    def report_results(self):
        pass


class Neuron:

    def __init__(self, vector_size):
        self.threshold = .5
        self.weights = [0 for x in range(vector_size+1)]
        self.weights[-1] = 1

    def _dot_product(self, vector, weights):
        """ Returns the dot product of two equal length vectors

        Args:
            vector (list)
            weights(list)

        Returns:
            a float
        """
        return sum(elem * weight for elem, weight in zip(vector, weights))

    def _sigmoid(self, z):
        return 1 / (1 + e ** (-z))

    def update_weights(self):
        pass

    def apply_backprop(self):
        pass

    def send_backprop(self):
        pass

    def fires(self):
        pass
        

def append_bias(vector):
    """ Takes a list of n entries and appends a 1 for the bias

    Args:
        vector - a list

    Returns:
        a list
    """
    temp_vector = [x for x in vector]
    temp_vector.append(1)
    return temp_vector
