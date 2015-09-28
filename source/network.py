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
        pass

    def run_unseen(self):
        pass

    def report_results(self):
        pass


class Neuron:

    def __init__(self, vector_size):
        self.threshold = threshold
        self.weights = [0 fo x in vector_size+1]
        self.weights[-1] = 1

    def _append_bias(self):
        pass

    def _dot_product(self):
        pass

    def _sigmoid(self):
        pass

    def update_weights(self):
        pass

    def apply_backprop(self):
        pass

    def send_backprop(self):
        pass

    def fires(self):
        pass


