import numpy as np

from .Base import BaseLayer

class Sigmoid(BaseLayer):
    def __init__(self):
        super().__init__()
        self.out= None

    def forward(self, input_tensor):
        self.out = 1 / (1 + np.exp(- input_tensor))
        return self.out

    def backward(self, error_tensor):
        return (self.out * (1 - self.out)) * error_tensor