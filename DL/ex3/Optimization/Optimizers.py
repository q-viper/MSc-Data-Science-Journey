import numpy as np

class Optimizer:
    def __init__(self, learning_rate= None):
        self.regularizer = None

    def add_regularizer(self, regularizer):
        self.regularizer = regularizer

class Sgd(Optimizer):
    def __init__(self, learning_rate: float):
        super().__init__()
        self.learning_rate = learning_rate

    def calculate_update(self, weight_tensor, gradient_tensor):
        updated_weight = weight_tensor-self.learning_rate*gradient_tensor
        if self.regularizer is not None:
            updated_weight = updated_weight - self.learning_rate*self.regularizer.calculate_gradient(weight_tensor)

        return updated_weight
    

class SgdWithMomentum(Optimizer):
    def __init__(self, learning_rate: float, momentum_rate:float):
        super().__init__()
        self.learning_rate = learning_rate
        self.momentum_rate=momentum_rate
        self.prev_gradient = 0


    def calculate_update(self, weight_tensor, gradient_tensor):
        self.prev_gradient = self.prev_gradient*self.momentum_rate - self.learning_rate*gradient_tensor
        updated_weight = self.prev_gradient+weight_tensor
        if self.regularizer is not None:
            updated_weight = updated_weight - self.learning_rate*self.regularizer.calculate_gradient(weight_tensor)
            
        return updated_weight
    
class Adam(Optimizer):
    def __init__(self, learning_rate: float, mu:float, rho: float):
        super().__init__()
        self.learning_rate = learning_rate
        self.mu=mu
        self.rho=rho

        self.vk = 0
        self.rk=0
        self.k =1


    def calculate_update(self, weight_tensor, gradient_tensor):
        gk = gradient_tensor
        self.vk = self.mu * self.vk + (1-self.mu) * gk
        self.rk = self.rho * self.rk + (1-self.rho) * (gk**2)
        rcap = self.rk/(1-self.rho**self.k)
        vcap = self.vk/(1-self.mu**self.k)
        self.k+=1
        gradient_tensor = vcap/(rcap**0.5 + np.finfo(float).eps)
        updated_weight = weight_tensor-self.learning_rate*gradient_tensor
        if self.regularizer is not None:
            updated_weight = updated_weight - self.learning_rate*self.regularizer.calculate_gradient(weight_tensor)
            
        return updated_weight