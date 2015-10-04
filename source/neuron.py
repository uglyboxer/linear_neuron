# Author: Cole Howard
#
# neuron.py is a basic linear neuron, that can be used in a perceptron
# Information on that can be found at:
# https://en.wikipedia.org/wiki/Perceptron
#
# It was written as a class specifically for network ()
#
# Usage:
#   From any python script:
#
#       from neuron import Neuron
#
# API:
#   update_weights, fires are the accessible methods
#   usage noted in their definitions


from math import e
from numpy import append as app
from numpy import dot


class Neuron:
    """ A class model for a single neuron

    Parameters
    ----------
    vector_size : int
        Length of an input vector
    target : int
        What the vector will associate with its weights.  It will claim this
        is the correct answer if it fires
    sample_size : int
        Total size of sample to be trained on
    answer_set: list
        The list of correct answers associated with the training set

    Attributes
    ----------
    threshold : float
        The tipping point at which the neuron fires (speifically in relation
        to the dot product of the sample vector and the weight set)
    weights : list
        The "storage" of the neuron.  These are changed with each training
        case and then used to determine if new cases will cause the neuron
        to fire.  The last entry is initialized to 1 as the weight of the
        bias
    expected : list
        Either 0's or 1's based on whether this neuron should for each of the
        vectors in the training set
    guesses : list
        Initialized to 0, and then updated with each training vector that
        comes through.
    """

    def __init__(self, vector_size, target, sample_size, answer_set):

        self.threshold = .5
        self.answer_set = answer_set
        self.target = target
        self.weights = [0 for x in range(vector_size + 1)]
        self.weights[-1] = 1    # Bias weight
        self.sample_size = sample_size
        self.expected = [0 if y != self.target else 1 for y in self.answer_set]
        self.guesses = [0 for z in range(self.sample_size)]

    def train_pass(self, vector, idx):
        """ Passes a vector through the neuron once

        Parameters
        ----------
        vector : a list
            Training vector
        idx : an int
            The position of the vector in the sample

        Returns
        -------
        None, always
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

    def _dot_product(self, vector):
        """ Returns the dot product of two equal length vectors

        Parameters
        ----------
        vector : list
            Any sample vector

        Returns
        -------
        float
            The sum for all of each element of a vector multiplied by its
            corresponding element in a second vector.
        """
        if len(vector) < len(self.weights):
            vector = app(vector, 1)
        return dot(vector, self.weights)

    def _sigmoid(self, z):
        """ Calculates the output of a logistic function

        Parameters
        ----------
        z : float
            The dot product of a sample vector and an associated weights
            set

        Returns
        -------
        float
            It will return something between 0 and 1 inclusive
        """

        if -700 < z < 700:
            return 1 / (1 + e ** (-z))
        elif z < -700:
            return 0
        else:
            return 1

    def update_weights(self, error, vector):
        """ Updates the weights stored in the receptors

        Parameters
        ----------
        error : int
            The distance from the expected value of a particular training
            case
        vector : list
            A sample vector

        Attributes
        ----------
        l_rate : float
            A number between 0 and 1, it will modify the error to control
            how much each weight is adjusted. Higher numbers will
            train faster (but risk unresolvable oscillations), lower
            numbers will train slower but be more stable.

        Returns
        -------
        None
        """

        l_rate = .05
        for idx, item in enumerate(vector):
            self.weights[idx] += (item * l_rate * error)

    def fires(self, vector):
        """ Takes an input vector and decides if neuron fires or not

        Parameters
        ----------
        vector : list
            A sample vector

        Returns
        -------
        bool
            Did it fire? True(yes) or False(no)
        float
            The dot product of the vector and weights
        """

        dp = self._dot_product(vector)
        if self._sigmoid(dp) > self.threshold:

            return True, dp
        else:
            return False, dp
