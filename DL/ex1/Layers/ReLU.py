from Layers.Base import *

class ReLU(BaseLayer):
    def __init__(self):
        super().__init__()

    def forward(self, input_tensor):
        self.input = input_tensor
        input_tensor[input_tensor<0]=0
        return input_tensor
    
    def backward(self, error_tensor):
        error_tensor[self.input<=0]=0
        # error_tensor[error_tensor>0]=1
        return error_tensor
