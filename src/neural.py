import numpy as np


class Neuron(object):

    def __init__(self, input_size=None, activation=None, weights=None, threshold=None):

        self.weights = np.random.normal(np.zeros([input_size]),1) if weights is None else weights
        self.threshold = np.random.normal(0,1) if threshold is None else threshold

        if activation is None:
            #self.activation = lambda x: 0 if (1 / (1 + np.exp(-x))) < self.threshold else 1
            self.activation = lambda x: 0 if x < 0 else 1
        else:
            self.activation = activation

    def stimulate(self, entry):
        if not isinstance(entry, np.ndarray):
            entry = np.array(entry)

        assert entry.shape == self.weights.shape

        return self.activation(np.dot(self.weights, entry) + self.threshold)
        #return self.activation(np.dot(self.weights, entry))

