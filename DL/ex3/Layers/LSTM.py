import copy
import numpy as np


from .Base import BaseLayer
from .FullyConnected import FullyConnected
from .TanH import TanH
from .Sigmoid import Sigmoid

class LSTM(BaseLayer):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.trainable = True
        self.regularization_loss = None

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.hidden_state = np.zeros(hidden_size)
        self.cell_state = np.zeros(hidden_size)

        self._memorize = False
        self._optimizer = None
        self._gradient_weights = None

        # LSTM
        # sigmoid(linear(input, hidden state)) -> forget gate i.e. ft, 1FCL, 1Sigmoid
        # forget gate output is vectorwise multiplied with hidden cell state i.e. ct
        # sigmoid(linear(input, hidden state)) -> input gate, 1FCL, 1Sigmoid
        # input gate output is vectorwise multiplied with tanh(linear(input, hidden state)) -> new_inp, 1FCL, 1TanH
        # new cell state = updated cell state + (forget gate output * new_inp)
        # sigmoid(linear(input, hidden) -> output gate, 1FCL, 1Sigmoid
        # output gate output is vectorwise multiplied with tanh(new cell state) -> new hidden state, 1TanH
        # output = sigmoid(hidden_state), 1Sigmoid
        # 
        self.forget_gate_fcl = FullyConnected(self.input_size + self.hidden_size, self.hidden_size)
        self.forget_gate_sig =  Sigmoid()
        self.input_gate_fcl = FullyConnected(self.input_size + self.hidden_size, self.hidden_size)
        self.input_gate_sig = Sigmoid()
        self.new_inp_fcl = FullyConnected(self.input_size + self.hidden_size, self.hidden_size)
        self.new_inp_tanh = TanH()
        self.output_gate_fcl = FullyConnected(self.input_size + self.hidden_size, self.hidden_size)
        self.output_gate_sig = Sigmoid()
        self.output_layer_sig = Sigmoid()
        self.output_layer_tanh = TanH()
        
        self.forget_gate_fcl_input_tensor = []
        self.forget_gate_sig_input_tensor = []
        self.input_gate_fcl_input_tensor = []
        self.input_gate_sig_input_tensor = []
        self.new_inp_fcl_input_tensor = []
        self.new_inp_tanh_input_tensor = []
        self.output_gate_fcl_input_tensor = []
        self.output_gate_sig_input_tensor = []
        self.output_layer_sig_input_tensor = []
        self.output_layer_tanh_input_tensor = []
        
        self.forget_gate_fcl_outputs = []
        self.forget_gate_sig_outputs = []
        self.input_gate_fcl_outputs = []
        self.input_gate_sig_outputs = []
        self.new_inp_fcl_outputs = []
        self.new_inp_tanh_outputs = []
        self.output_gate_fcl_outputs = []
        self.output_gate_sig_outputs = []
        self.output_layer_sig_outputs = []
        self.output_layer_tanh_outputs = []

        
        self.forget_gate_fcl_gradient_weights = []
        self.input_gate_fcl_gradient_weights = []
        self.new_inp_fcl_gradient_weights = []
        self.output_gate_fcl_gradient_weights = []


    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, value):
        self._optimizer = value

    @property
    def memorize(self):
        return self._memorize

    @memorize.setter
    def memorize(self, value):
        self._memorize = value

    # @property
    # def weights(self):
    #     return self.hidden_fcl.weights

    # @weights.setter
    # def weights(self, value):
    #     self.hidden_fcl.weights = value

    # @property
    # def gradient_weights(self):
    #     return self._gradient_weights

    # @gradient_weights.setter
    # def gradient_weights(self, value):
    #     self._gradient_weights = value

    def calculate_regularization_loss(self):
        if self.optimizer.regularizer:
            self.regularization_loss += self.optimizer.regularizer.norm(self.forget_gate_fcl.weights)
            self.regularization_loss += self.optimizer.regularizer.norm(self.input_gate_fcl.weights)
            self.regularization_loss += self.optimizer.regularizer.norm(self.new_inp_fcl.weights)
            self.regularization_loss += self.optimizer.regularizer.norm(self.output_gate_fcl.weights)

            # self.regularization_loss += self.optimizer.regularizer.norm(self.hidden_fcl.weights)
            # self.regularization_loss += self.optimizer.regularizer.norm(self.output_fcl.weights)
        return self.regularization_loss
        

    def initialize(self, weights_initializer, bias_initializer):
        self.forget_gate_fcl.initialize(weights_initializer, bias_initializer)
        self.input_gate_fcl.initialize(weights_initializer, bias_initializer)
        self.new_inp_fcl.initialize(weights_initializer, bias_initializer)
        self.output_gate_fcl.initialize(weights_initializer, bias_initializer)
        
    def forward(self, input_tensor):
        
        previous_hstate = self.hidden_state if self.memorize else np.zeros(self.hidden_size)
        previous_cstate = self.cell_state if self.memorize else np.zeros(self.hidden_size)

        batch_size,*_ = input_tensor.shape
        output_tensor = np.zeros((batch_size,self.output_size))

        # for each batch dimension or time dimension?
        for i in range(input_tensor.shape[0]):
            xcombined = np.concatenate([previous_hstate, input_tensor[i]]).reshape(1, -1)
            
            # forget gate
            ft = self.forget_gate_fcl.forward(xcombined)
            ft = self.forget_gate_sig.forward(ft)

            # input gate
            it = self.input_gate_fcl.forward(xcombined)
            it = self.input_gate_sig.forward(it)

            # new input
            new_inp = self.new_inp_fcl.forward(xcombined)
            new_inp = self.new_inp_tanh.forward(new_inp)

            # output gate
            ot = self.output_gate_fcl.forward(xcombined)
            ot = self.output_gate_sig.forward(ot)

            oth = self.output_layer_tanh.forward(current_cstate)

            # current cell state 
            current_cstate = ft * previous_cstate + it * new_inp
            
            # current hidden state
            current_hstate = ot * oth

            # output
            output = self.output_layer_sig.forward(current_hstate)
            output_tensor[i] = output[0]

            # store inputs and outputs for backprop
            self.forget_gate_input_tensor.append(self.forget_gate_fcl.input_tensor)
            self.input_gate_input_tensor.append(self.input_gate_fcl.input_tensor)
            self.new_inp_input_tensor.append(self.new_inp_fcl.input_tensor)
            self.output_gate_input_tensor.append(self.output_gate_fcl.input_tensor)
            self.output_layer_input_tensor.append(self.output_layer_fcl.input_tensor)

            self.forget_gate_fcl_outputs.append(ft)
            self.input_gate_fcl_outputs.append(it)
            self.new_inp_fcl_outputs.append(new_inp)
            self.output_gate_fcl_outputs.append(ot)
            self.output_layer_tanh_outputs.append(oth)   
            self.output_layer_sig_outputs.append(output)

        # update hidden state
        self.hidden_state = current_hstate[0]

        # update cell state
        self.cell_state = current_cstate[0]

        return output_tensor

    def backward(self, error_tensor):
        # get gradient wts for hidden fcl and output fcl
        self.gradient_weights = np.zeros_like(self.hidden_fcl.weights)
        self.output_fcl_gradient_weights = np.zeros_like(self.output_fcl.weights)
        
        gradient_prev_hstate = 0
        batch_size,*_ = error_tensor.shape
        gradient_wrt_inputs = np.zeros((batch_size,self.input_size))

        # for each time dimension, we need to calculate the gradient
        # and sum them up just to make it look like using memory to update :P
        for step in range(error_tensor.shape[0] - 1, -1, -1):
            
            self.output_layer_sig.out = self.output_layer_sig_outputs[step]
            output_layer_sig_error = self.output_layer_sig.backward(error_tensor[step])

            self.output_layer_tanh.out = self.output_layer_tanh_outputs[step]
            output_layer_tanh_error = self.output_layer_tanh.backward(output_layer_sig_error)

            self.output_gate_sig.out = self.output_gate_sig_outputs[step]
            output_gate_sig_error = self.output_gate_sig.backward(output_layer_sig_error)

            self.output_gate_fcl.out = self.output_gate_fcl_outputs[step]
            output_gate_fcl_error = self.output_gate_fcl.backward(output_gate_sig_error)

            self.new_inp_tanh.out = self.new_inp_tanh_outputs[step]
            new_inp_tanh_error = self.new_inp_tanh.backward(output_gate_fcl_error)

            self.new_inp_fcl.out = self.new_inp_fcl_outputs[step]
            new_inp_fcl_error = self.new_inp_fcl.backward(new_inp_tanh_error)

            self.input_gate_sig.out = self.input_gate_sig_outputs[step]
            input_gate_sig_error = self.input_gate_sig.backward(new_inp_fcl_error)

            self.input_gate_fcl.out = self.input_gate_fcl_outputs[step]
            input_gate_fcl_error = self.input_gate_fcl.backward(input_gate_sig_error)

            self.forget_gate_sig.out = self.forget_gate_sig_outputs[step]
            forget_gate_sig_error = self.forget_gate_sig.backward(input_gate_fcl_error)

            self.forget_gate_fcl.out = self.forget_gate_fcl_outputs[step]
            forget_gate_fcl_error = self.forget_gate_fcl.backward(forget_gate_sig_error)
            

            

            # split the error in hidden layer into two parts
            # one part is for the hidden state and the other is for the input
            gradient_prev_hstate = forget_gate_fcl_error[:, :self.hidden_size]
            gradient_with_respect_to_input = forget_gate_fcl_error[:, self.hidden_size:]
            gradient_wrt_inputs[step]=gradient_with_respect_to_input[0]

            # sum up the gradient weights for hidden fcl and output fcl
            # this is what used while updating the weights
            # the gradients are not only dependent on the current step
            # but also on the previous (batch_size) steps
            self.gradient_weights+=self.hidden_fcl.gradient_weights
            self.output_fcl_gradient_weights+=self.output_fcl.gradient_weights


        if self.optimizer:
            self.output_fcl.weights = self.optimizer.calculate_update(self.output_fcl.weights,
                                                                           self.output_fcl_gradient_weights)
            self.weights = self.optimizer.calculate_update(self.weights, self.gradient_weights)

        return gradient_wrt_inputs

    