import numpy as np
from copy import deepcopy

from Layers.Conv import Conv
from Layers.Flatten import Flatten
from Layers.FullyConnected import FullyConnected
from Layers.Pooling import Pooling
from Layers.ReLU import ReLU
from Layers.SoftMax import SoftMax
from Optimization.Optimizers import Adam
from Optimization.Constraints import L2_Regularizer
from NeuralNetwork import NeuralNetwork
from Layers.Initializers import Xavier, He, Constant
from Optimization.Loss import CrossEntropyLoss

class LeNet(NeuralNetwork):
    def __init__(self, optimizer, weight_initializer, bias_initializer):
        super().__init__(optimizer, weight_initializer, bias_initializer)

        self.optimizer.add_regularizer(L2_Regularizer(0.0004))
        self.loss_layer=CrossEntropyLoss()

def build():
    LENET = LeNet(optimizer=Adam(learning_rate=0.0005, mu=0.9, rho=0.999), 
            weight_initializer=Xavier(), 
            bias_initializer=Constant(0.1))
            # 3 conv layers
    LENET.append_layer(Conv((1, 1), (1, 5, 5), 6))
    LENET.append_layer(ReLU())
    LENET.append_layer(Pooling((2, 2), (2, 2)))
    LENET.append_layer(Conv((1, 1), (6, 5, 5), 16)) # recheck 1 or 
    LENET.append_layer(ReLU())
    LENET.append_layer(Pooling((2, 2), (2, 2)))
    LENET.append_layer(Flatten())
    LENET.append_layer(FullyConnected(16 * 7 * 7, 120))
    LENET.append_layer(ReLU())
    LENET.append_layer(FullyConnected(120, 84))
    LENET.append_layer(ReLU())
    LENET.append_layer(FullyConnected(84, 10))
    LENET.append_layer(SoftMax())

    return LENET

