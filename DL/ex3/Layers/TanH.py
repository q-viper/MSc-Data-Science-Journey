import numpy as np

from Layers.Base import *


class TanH(BaseLayer):
    def __init__(self):
        super().__init__()
        self.out = None
    
    def forward(self, input_tensor):
        self.out = np.tanh(input_tensor)
        return self.out
    
    def backward(self, error_tensor):
         return (1 - self.out**2) * error_tensor

