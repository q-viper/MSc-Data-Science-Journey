import numpy as np

class Constant:
    def __init__(self, const=0.1):
        self.const=const
    def  initialize(self, weights_shape, fan_in, fan_out):
        self.tensor = np.zeros(weights_shape)+self.const
        return self.tensor

class UniformRandom:
    def  initialize(self, weights_shape, fan_in, fan_out):
        self.tensor = np.random.uniform(size=(weights_shape[0], weights_shape[1]))
        return self.tensor

class Xavier:
    def  initialize(self, weights_shape, fan_in, fan_out):
        self.tensor = (2/(fan_in+fan_out))**0.5
        self.tensor = np.random.normal(0, self.tensor, weights_shape)
        return self.tensor

class He:
    def  initialize(self, weights_shape, fan_in, fan_out):
        self.tensor = (2/fan_in)**0.5
        self.tensor = np.random.normal(0, self.tensor, weights_shape)
        return self.tensor