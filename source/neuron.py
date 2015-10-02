from math import e


class Neuron:

    def __init__(self, vector_size, target, sample_size, answer_set):
        """ A class model for a single neuron

        Args:
            vector_size: int
            target: int
            sample_size: int
            answer_set: listhttps://gsnedders.html5.org/outliner/
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
        if self._sigmoid(dp) > self.threshold:

            return True, dp
        else:
            return False, dp
