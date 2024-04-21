import copy
import numpy as np


from .Base import BaseLayer
from .FullyConnected import FullyConnected
from .TanH import TanH
from .Sigmoid import Sigmoid

class RNN(BaseLayer):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.trainable = True
        self.regularization_loss = None

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.hidden_state = np.zeros(hidden_size)

        self._memorize = False
        self._optimizer = None
        self._gradient_weights = None

        # Elman network has tanh activation function and 
        # sigmoid activation function (to add non-linearity to output) 
        # in hidden layers
        # input - > Hidden(FullyConnected -> tanh ->  FullyConnected-> sigmoid ) -> Output
        # Output of Hidden is sent as input to the Hidden layer in next step (some sort of memory)
        # hidden fcl will accept input of size (input_size + hidden_size)
        # where input size is example dataset size and 
        # hidden_size is the size of output of hidden layer i.e. hidden_size

        self.tanh = TanH()
        self.sigmoid = Sigmoid()
        self.hidden_fcl = FullyConnected(self.input_size + self.hidden_size, self.hidden_size)
        self.output_fcl = FullyConnected(self.hidden_size, self.output_size)

        self.hidden_fcl_input_tensor = np.ndarray([])
        self.output_fcl_input_tensors = []
        self.sigmoid_outputs = []
        self.tanh_outputs = []

        self.hidden_fcl_gradient_weights = []
        self.output_fcl_gradient_weights = []
        
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

    @property
    def weights(self):
        return self.hidden_fcl.weights

    @weights.setter
    def weights(self, value):
        self.hidden_fcl.weights = value

    @property
    def gradient_weights(self):
        return self._gradient_weights

    @gradient_weights.setter
    def gradient_weights(self, value):
        self._gradient_weights = value

    def calculate_regularization_loss(self):
        if self.optimizer.regularizer:
            self.regularization_loss += self.optimizer.regularizer.norm(self.hidden_fcl.weights)
            self.regularization_loss += self.optimizer.regularizer.norm(self.output_fcl.weights)
        return self.regularization_loss
        

    def initialize(self, weights_initializer, bias_initializer):
        self.hidden_fcl.initialize(weights_initializer, bias_initializer)
        self.output_fcl.initialize(weights_initializer, bias_initializer)

    def forward(self, input_tensor):
        self.sigmoid_outputs = []
        self.tanh_outputs = []
        self.output_fc_input_tensors = []
        self.hidden_fcl_input_tensors = []
        previous_hstate = self.hidden_state if self.memorize else np.zeros(self.hidden_size)
        batch_size,*_ = input_tensor.shape
        output_tensor = np.zeros((batch_size,self.output_size))

        # for each batch dimension or time dimension?
        for i in range(input_tensor.shape[0]):
            # for first step, input will be input_tensor and 0
            # for next steps, input will be input_tensor and previous hidden state
            # use the hidden state as input to the hidden layer
            new_input = np.concatenate([previous_hstate, input_tensor[i]]).reshape(1, -1)

            input_to_tanh = self.hidden_fcl.forward(new_input)
            # current hidden state will be output of tanh
            # this is the hidden state that will be used in next step
            tanh_output = self.tanh.forward(input_to_tanh)
            current_hstate = tanh_output
            previous_hstate = tanh_output[0]

            # use the tanh output as input to the FC layer before sigmoid layer 
            input_to_sigmoid = self.output_fcl.forward(tanh_output)
            sigmoid_output = self.sigmoid.forward(input_to_sigmoid)

            output_tensor[i]=sigmoid_output[0]

            # store inputs and outputs for backprop
            self.hidden_fcl_input_tensors.append(self.hidden_fcl.input_tensor)
            self.output_fc_input_tensors.append(self.output_fcl.input_tensor)
            self.sigmoid_outputs.append(self.sigmoid.out)
            self.tanh_outputs.append(self.tanh.out)

        # update hidden state
        self.hidden_state = current_hstate[0]

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
            # what was the output of sigmoid layer at step `step`? 
            # we need to use that to calculate the error in this layer at this step
            self.sigmoid.out = self.sigmoid_outputs[step]
            sigmoid_error = self.sigmoid.backward(error_tensor[step])

            # what was the input to the output layer at step `step`?
            # since sigmoid is last, we pass sigmoid error backward to this layer
            # but again it should use output at that step to calculate error at that step
            self.output_fcl.input_tensor = self.output_fc_input_tensors[step]
            output_fcl_error = self.output_fcl.backward(sigmoid_error)

            # same as above two steps except, 
            # we need to add the gradient from previous hidden state
            # this is where the memory comes in while updating the weights 
            self.tanh.out = self.tanh_outputs[step]
            tanh_error = self.tanh.backward(output_fcl_error + gradient_prev_hstate)

            # pass error in tanh to the hidden layer now
            self.hidden_fcl.input_tensor = self.hidden_fcl_input_tensors[step]
            hidden_fcl_error = self.hidden_fcl.backward(tanh_error)

            # split the error in hidden layer into two parts
            # one part is for the hidden state and the other is for the input
            gradient_prev_hstate = hidden_fcl_error[:, :self.hidden_size]
            gradient_with_respect_to_input = hidden_fcl_error[:, self.hidden_size:]
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

    