import numpy as np

from Layers.Base import *


class Dropout(BaseLayer):

    def __init__(self, probability):
        super().__init__()

        self.prob = probability
        self.mask = None

    def forward(self, input_tensor):
        # if testing no need to
        if self.testing_phase:
            return input_tensor

        self.mask = np.random.rand(*input_tensor.shape) < self.prob
        return (input_tensor*self.mask) / self.prob


    def backward(self, error_tensor):

        return (self.mask*error_tensor  ) / self.prob


