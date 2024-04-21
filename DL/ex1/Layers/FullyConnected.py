import numpy as np

from Layers.Base import *
from Optimization.Optimizers import *

class FullyConnected(BaseLayer):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.uniform(size=(input_size, output_size))
        self.bias = np.ones(self.output_size)
        self.trainable=True
        self._optimizer = None
        self.gradient_tensor=None


    @property
    def optimizer(self):
        return self._optimizer
    
    @optimizer.setter
    def optimizer(self, opt):
        self._optimizer = opt

    def forward(self, input_tensor):
        # print(input_tensor.shape, self.weights.shape)
        self.input = input_tensor
        out = np.dot(input_tensor, self.weights)+self.bias
        self.out = out
        return out

    def backward(self, error_tensor):
        # print(self.input.shape, self.out.shape, error_tensor.shape, self.weights.shape)
        self.error_tensor = np.dot(error_tensor,self.weights.T)
        self._gradient_tensor = np.dot(np.atleast_2d(self.input).T,error_tensor)
        
        # print(self.bias.shape, error_tensor.shape)

        if self.optimizer is not None:
            self.weights = self.optimizer.calculate_update(self.weights, self._gradient_tensor)
            self.bias = self.optimizer.calculate_update(self.bias, error_tensor.sum(axis=0))
        return self.error_tensor
    
    @property
    def gradient_weights(self):
        # self.gradient_tensor=None
        return self._gradient_tensor
    



