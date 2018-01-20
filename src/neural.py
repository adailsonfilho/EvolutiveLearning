import numpy as np


class Neuron(object):

    def __init__(self, input_size=None, activation=None, weights=None, threshold=None):

        self.weights = np.random.random([input_size]) if weights is None else weights
        self.threshold = np.random.random() if threshold is None else threshold

        if activation is None:
            self.activation = lambda x: 0 if (1 / (1 + np.exp(-x))) < self.threshold else 1
        else:
            self.activation = activation

    def stimulate(self, entry):
        if not isinstance(entry, np.ndarray):
            entry = np.array(entry)

        assert entry.shape == self.weights.shape

        return self.activation(np.dot(entry, self.weights))
