# Author: Cole Howard
#
# network.py is a basic implementation of a one layer neural network, to
# examine an implementation of backpropagation.  The intent is to make it
# extensible as a tool for exploring neural nets under more general
# circumstances.

from math import e
from random import choice

from sklearn import datasets


class Network:

    def __init__(self, neuron_targets, vector_size, train_set, train_answers,
                 test_set, test_answers, epoch_nums=100, layers=1,
                 learn_iter=1):
        """ A Network instance will create layers of neurons for the implementa-
        tion of neural network.

        Args:
            neuron_targets: list
            vector_size: int (training or test)
            train_set: a list or numpy array
            train_answers: a list or numpy array
            test_set: a list or numpy array
            test_answers: a list or numpy array
            epoch_nums: an int (the number of times backprop should occur)
            layers: int
            learn_iter: an int (the number of times before backprop)
        """

        self.neuron_count = neuron_targets    # Per layer
        self.vector_size = vector_size
        self.train_set = train_set
        self.train_answers = train_answers
        self.test_set = test_set
        self.test_answers = test_answers
        self.epoch_nums = epoch_nums
        self.layers = layers
        self.learn_iter = learn_iter        # Size of leaning sets/iteration
        self.neurons = [Neuron(self.vector_size, x, len(self.train_set),
                        self.train_answers) for x in self.neuron_count]
        self.predictions = []

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

    def gradient_descent(self, vector, vector_index):
        """ Calculates the gradient_descent

        Args:
            vector: a list
            vector_index: an int

        Returns:
            an int
        """
        learning_rate = .05
        temp_list = [(self.neurons[x]._sigmoid(self.neurons[x]._dot_product(vector)) - self.neurons[x].expected[vector_index]) * self.neurons[x]._sigmoid(self.neurons[x]._dot_product(vector)) * (1 - self.neurons[x]._sigmoid(self.neurons[x]._dot_product(vector))) for x in self.neuron_count]
        gd = -1 * learning_rate * sum(temp_list)
        return gd

    def learn_run(self):
        """ Runs an iteration through the neuron sets and adjust the weights
        appropriately.
        """
        for idx, vector in enumerate(self.train_set):
            for neuron in self.neurons:
                neuron.train_pass(vector, idx)
            gd = self.gradient_descent(vector, idx)    # Backpropogate the error
            for neuron in self.neurons:
                neuron.update_weights(gd, vector)

    def run_unseen(self):
        """ Makes guesses on the unseen data

        Returns:
            a list of ints (the guesses for each vector)

        """

        temp_guess_list = [[] for x in self.test_set]
        temp_dud_guess_list = [[] for x in self.test_set]
        for idy, vector in enumerate(self.test_set):

            for idx, neuron in enumerate(self.neurons):
                nf = neuron.fires(vector)
                if nf[0]:
                    temp_guess_list[idy].append((nf[1], idx))
                    temp_dud_guess_list[idy].append((0, idx))
                else:
                    temp_guess_list[idy].append((0, None))
                    temp_dud_guess_list[idy].append((nf[1], idx))
            temp_guess_list[idy].sort(reverse=True)
            temp_dud_guess_list[idy].sort(reverse=True)
        guess_list = [x[0][1] for x in temp_guess_list]
        dud_guess_list = [x[0][1] for x in temp_dud_guess_list]
        new_guess_list = [x if (x != None) else dud_guess_list[idx] for
                          idx, x in enumerate(guess_list)]
        return new_guess_list

    def report_results(self, guess_list):
        """ Reports results of guesses on unseen set

        Args:
            guess_list: a list 
        """
        successes = 0
        for idx, item in enumerate(guess_list):
            if self.test_answers[idx] == item:
                successes += 1
        print("Successes: {}  Out of total: {}".format(successes, len(guess_list)))
        print("For a success rate of: ", successes/len(guess_list))


class Neuron:

    def __init__(self, vector_size, target, sample_size, answer_set):
        """ A class model for a single neuron

        Args:
            vector_size: int
            target: int
            sample_size: int
            answer_set: list
        """
        self.threshold = .5
        self.answer_set = answer_set
        self.target = target
        self.weights = [0 for x in range(vector_size + 1)]
        self.weights[-1] = 1
        self.sample_size = sample_size
        self.expected = [0 if y != self.target else 1 for y in self.answer_set]
        self.guesses = [0 for z in range(self.sample_size)]

    def train_pass(self, vector, idx):
        """ Passes a vector through the neuron once

        Args:
            vector: a list
            idx: an int         # The position of the vector in the sample
        """
        if self.expected == self.guesses:
            return None
        else:
            error = self.expected[idx] - self.guesses[idx]
            if self.fires(vector)[0]:
                self.guesses[idx] = 1
            else:
                self.guesses[idx] = 0
            self.update_weights(error, vector)
            return None
#### Look at this again

    def _dot_product(self, vector):
        """ Returns the dot product of two equal length vectors

        Args:
            vector: a list

        Returns:
            a float
        """
        return sum(elem * weight for elem, weight in zip(vector, self.weights))

    def _sigmoid(self, z):
        if -700 < z < 700:
            return 1 / (1 + e ** (-z))
        elif z < -700:
            return 0
        else:
            return 1

    def update_weights(self, error, vector):
        """ Updates the weights stored in the receptors

        Args:
            error (int)
            vector(list)
        Returns:
            None
        """
        l_rate = .05
        for idx, item in enumerate(vector):
            self.weights[idx] += (item * l_rate * error)

    def apply_backprop(self):
        pass

    def send_backprop(self):
        pass

    def fires(self, vector):
        """ Takes an input vector and decides if neuron fires or not

        Args:
            vector - a list

        Returns:
            a boolean
            a float             # The dot product of the vector and weights
        """
        dp = self._dot_product(vector)
        if dp > self.threshold:

            return True, dp
        else:
            return False, dp


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


def main():
    # Dependent on input set
    digits = datasets.load_digits()
    target_values = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    num_of_training_vectors = 950
    answers, answers_to_test = digits.target[:num_of_training_vectors], digits.target[num_of_training_vectors:]
    training_set, testing_set = digits.data[:num_of_training_vectors], digits.data[num_of_training_vectors:]

    # For all inputs
    training_vectors = [append_bias(vector) for vector in training_set]
    test_vectors = [append_bias(vector) for vector in testing_set]

    network = Network(target_values, len(training_set[0]), training_vectors,
                      answers, test_vectors, answers_to_test)
    [network.learn_run() for x in range(250)]
    network.report_results(network.run_unseen())

if __name__ == '__main__':
    main()
