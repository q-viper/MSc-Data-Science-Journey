import numpy as np

from Layers.Base import *

class SoftMax(BaseLayer):
    def __init__(self):
        super().__init__()

    def forward(self, input_tensor):
        self.input = input_tensor-input_tensor.max(axis=1)[:, np.newaxis]
        e = np.exp(self.input)
        # print("e ", e.shape)
        out = e/e.sum(axis=1)[:, np.newaxis]
        # print("o", out.shape)
        self.out = out
        return out

    def backward(self, error_tensor):
        # print(self.output.shape, error_tensor.shape)
        # et = self.output*(error_tensor-np.dot(error_tensor))
        et = error_tensor*self.out #[:, np.newaxis]
        et = et.sum(axis=1)
        et = error_tensor-et[:, np.newaxis]
        et = self.out*et
        # print(error_tensor.shape, self.output.shape)
        return et