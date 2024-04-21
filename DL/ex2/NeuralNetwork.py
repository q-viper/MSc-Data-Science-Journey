from copy import deepcopy

class NeuralNetwork:
    def __init__(self, optimizer, weights_initializer, bias_initializer) -> None:
        self.optimizer = optimizer
        self.loss = []
        self.layers=[]
        self.data_layer = None
        self.loss_layer = None
        self.weights_initializer = weights_initializer
        self.bias_initializer = bias_initializer

    def forward(self):
        inp,op = self.data_layer.next()
        self.label = op
        # print(inp)
        for layer in self.layers:
            inp = layer.forward(inp)
        inp = self.loss_layer.forward(inp, self.label)
        self.pred=inp
        return self.pred
            
    
    def backward(self):
        # loss = self.loss_layer.forward(self.pred, self.label)
        loss = self.loss_layer.backward(self.label)
        for layer in self.layers[::-1]:
            loss = layer.backward(loss)

    
    def append_layer(self, layer):
        if layer.trainable:
            layer.optimizer = deepcopy(self.optimizer)
            layer.initialize(self.weights_initializer, self.bias_initializer)

        self.layers.append(layer)

    def train(self, iterations):
        for i in range(iterations):
            loss = self.forward()
            self.backward()
            self.loss.append(loss)

    def test(self, input_tensor):
        inp = input_tensor #self.data_layer.next()
        for layer in self.layers:
            inp = layer.forward(inp)
        # print(layer)
        return inp


