import numpy as np

from Layers.Base import *
from Layers.Initializers import *
from Optimization.Optimizers import *


class FullyConnected(BaseLayer):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.uniform(size=(input_size+1, output_size))
        # self.bias = np.ones(self.output_size)
        self.trainable=True
        self._optimizer = None
        self.gradient_tensor=None
        self.input_tensor=None
        self.gradient_weights=None

    @property
    def optimizer(self):
        return self._optimizer
    
    @optimizer.setter
    def optimizer(self, opt):
        self._optimizer = opt

    def initialize(self, weights_initializer, bias_initializer):
        weights = weights_initializer.initialize((self.input_size, self.output_size), self.input_size, self.output_size)
        bias = bias_initializer.initialize((1,self.output_size), 1, self.output_size)
        self.weights=np.vstack([weights, bias])


    def forward(self, input_tensor):
        # print(input_tensor.shape, self.weights.shape)
        self.input_tensor = np.column_stack((input_tensor,np.ones(input_tensor.shape[0])))
        out = np.dot(self.input_tensor, self.weights)
        self.out = out
        return out

    def backward(self, error_tensor):
        # print(self.input.shape, self.out.shape, error_tensor.shape, self.weights.shape)
        self.error_tensor = np.dot(error_tensor,np.delete(self.weights, -1, 0).T)
        gradient_tensor = np.dot(np.atleast_2d(self.input_tensor).T,error_tensor)
        self.gradient_weights = gradient_tensor

        if self.optimizer is not None:
            self.weights = self.optimizer.calculate_update(self.weights, gradient_tensor)
            # self.bias = self.optimizer.calculate_update(self.bias, error_tensor.sum(axis=0))
        return self.error_tensor
    

    



