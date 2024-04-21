from copy import deepcopy

def save(filename, net):
    import pickle
    nnet=net
    dlayer = nnet.data_layer
    nnet.__setstate__({'data_layer': None})

    with open(filename, 'wb') as f:
        pickle.dump(nnet, f)
    nnet.__setstate__({'data_layer': dlayer})
    

def load(filename, data_layer):
    import pickle
    with open(filename, 'rb') as f:
        net = pickle.load(f)
        net.__setstate__({'data_layer': data_layer})
        
    return net

class NeuralNetwork:
    def __init__(self, optimizer, weights_initializer, bias_initializer) -> None:
        self.optimizer = optimizer
        self.loss = []
        self.layers=[]
        self.data_layer = None
        self.loss_layer = None
        self.weights_initializer = weights_initializer
        self.bias_initializer = bias_initializer

        self._phase = None

    def __getstate__(self):
        return self.__dict__.copy()
    
    def __setstate__(self, state):
        self.__dict__.update(state)
        return self.__dict__.copy()

    @property
    def phase(self):
        return self._phase
    
    @phase.setter
    def phase(self, value):
        self._phase = value

    def forward(self):
        inp,op = self.data_layer.next()
        self.label = op
        regularization_loss = 0
        # print(inp)
        for layer in self.layers:
            inp = layer.forward(inp)
            try:
                regularization_loss += self.optimizer.regularizer.norm(layer.weights)
            except:
                pass        
            layer.testing_phase = True

        # inp = self.loss_layer.forward(inp, self.label)
        self.pred=self.loss_layer.forward(inp+regularization_loss, op)
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
        # print(inp.shape)
        for layer in self.layers:
            inp = layer.forward(inp)
        # print(layer)
        return inp


