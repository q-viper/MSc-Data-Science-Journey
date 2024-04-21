import numpy as np
from Layers.Base import *

class Flatten(BaseLayer):
    def __init__(self):
        super().__init__()
        self.input_tensor = None
        self.output_tensor = None
        
    def forward(self, input_tensor):
        self.input_tensor=input_tensor
        self.output_tensor = input_tensor.flatten().reshape(input_tensor.shape[0],-1)
        return self.output_tensor
    
    def backward(self, error_tensor):
        return error_tensor.reshape(self.input_tensor.shape)
