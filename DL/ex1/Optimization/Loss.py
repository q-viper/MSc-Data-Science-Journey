import numpy as np

class CrossEntropyLoss:
    def __init__(self):
        pass
    def forward(self,prediction_tensor, label_tensor):
        self.input = prediction_tensor
        l = - np.log(prediction_tensor+np.finfo(float).eps)
        l[label_tensor!=1]=0
        l = l.sum()
        return l
    def backward(self, label_tensor):
        error_tensor = -label_tensor/(self.input+np.finfo(float).eps)
        return error_tensor
