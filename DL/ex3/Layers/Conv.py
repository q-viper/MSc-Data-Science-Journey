from Layers.Base import BaseLayer

import numpy as np
from scipy import signal
from functools import reduce
import operator
from copy import deepcopy as copy


class Conv(BaseLayer):
    def __init__(self, stride_shape, convolution_shape, num_kernels):
        super().__init__()
        self.trainable = True
        self.stride_shape = (stride_shape[0], stride_shape[0]) if len(stride_shape) == 1 else stride_shape
        # 1d as [channel,m], 2d as [channel,m,n]
        self.convolution_shape = convolution_shape
        self.num_kernels = num_kernels
        # init weights as uniform random (will be initialized again with initialize method)
        # shape for 2d conv: (num_kernels, channel, m, n) 
        self.weights = np.random.uniform(0, 1, (num_kernels, *convolution_shape))
        # bias shape: number of kernels
        self.bias = np.random.rand(num_kernels) 
        
        # grad parameters
        self._gradient_weights = None
        self._gradient_bias = None

        self._optimizer = None
        self._bias_optimizer = None

        # conv_dim if it is 2d or 1d
        self.conv_dim = 2 if len(convolution_shape) == 3 else 1


    def initialize(self, weights_initializer, bias_initializer):
        self.weights = weights_initializer.initialize(self.weights.shape,
                    reduce(operator.mul, self.convolution_shape),
                    reduce(operator.mul, [self.num_kernels, *self.convolution_shape[1:]]))

        self.bias = bias_initializer.initialize(self.bias.shape, 1,self.num_kernels)

        self._optimizer=copy(self.optimizer)
        self._bias_optimizer=copy(self.optimizer)

    def forward(self, input_tensor):
        # if correlation is used in forward, we can use convole in backward
        # or vice versa
        # input_tensor shape (b,c,x,y) or (b,c,x)
        self.input_tensor = input_tensor
        ishape = input_tensor.shape
        self.ishape = ishape
        bsize, c, y, x = ishape if self.conv_dim==2 else (*ishape, None)
        cx,cy = self.convolution_shape[-2:]

        sh, sw = self.stride_shape

        # new shape of y = (y-ky + 2*p)/sh + 1; y input size, ky kernel size, p padding size, sh stride size
        #  but we need o/p size same as i/p so p=(ky-1)/2 if sh==1
        # else we need to derive
        pad=[(cx-1)/2]
        out_shape = [int((y-cx+2*pad[0])/sh)+1]
        if self.conv_dim==2:
            pad.append((cy-1)/2)
            out_shape.append(int((x-cy+2*pad[1])/sw)+1)
        self.pad=pad
        result = np.zeros((bsize, self.num_kernels, *out_shape))

        # if used correlation in forward, should use convolve in backward 
        for cb in range(bsize):
            for ck in range(self.num_kernels):
                # sum outputs of correlation of this kernel with individual input channel of input
                kout = np.zeros((y,x)) if x else np.zeros((y))
                for ch in range(c):
                    # correlate with this batch's this channel and this kernel's this channel
                    kout += signal.correlate(input_tensor[cb, ch], self.weights[ck, ch], mode='same', method='direct')
                  
                kout = kout[::sh, ::sw] if self.conv_dim==2 else kout[::sh]
                result[cb, ck] = kout + self.bias[ck]

        return result


    def update_parameters(self, error_tensor):
        # what is the grad of bias in this layer for this batch?
        # we sum error tensor along axis of B,W,H (if 2d)
        # B
        berror = error_tensor.sum(axis=0)
        # W
        yerror = berror.sum(axis=1)
        # H?
        self._gradient_bias = yerror.sum(axis=1) if self.conv_dim==2 else yerror

        # what is the grad of weights in this layer for this batch?
        batch_size, channels, y, x = self.ishape if self.conv_dim==2 else (*self.ishape, None)
        sh, sw = self.stride_shape
        cx, cy = self.convolution_shape[-2:]

        self.gradient_weights=np.zeros_like(self.weights)
        for cb in range(batch_size):
            for ch in range(channels):
                for ck in range(self.num_kernels):
                    if self.conv_dim==2:
                        error = np.zeros((y, x))
                        error[::sh, ::sw] = error_tensor[cb, ck]
                        inp = np.pad(self.input_tensor[cb, ch],
                                                    [(int(np.ceil(self.pad[0])), int(np.floor(self.pad[0]))), 
                                                    (int(np.ceil(self.pad[1])), int(np.floor(self.pad[1])))]
                                                    #  [int(np.ceil(self.pad[0])), int(np.floor(self.pad[1]))]
                                                     )
                    else:
                        error = np.zeros(y)
                        error[::sh] = error_tensor[cb, ck]
                        inp = np.pad(self.input_tensor[cb, ch], [(int(np.ceil(self.pad[0])), int(np.floor(self.pad[0])))])

                    self.gradient_weights[ck, ch] += signal.correlate(
                        inp, error, mode='valid')

        if self.optimizer:
            self.weights = self._optimizer.calculate_update(self.weights, self._gradient_weights)
            self.bias = self._bias_optimizer.calculate_update(self.bias, self._gradient_bias)

    def error_this_layer(self, error_tensor):
        # compute error in this layer
        gradient=np.zeros_like(self.input_tensor)
        sh,sw = self.stride_shape

        # input Conv2d weight shape: (num_kernel, channel, w, h), channel is channel of input data
        # inner Conv2d weight shape: (num_kernel, input_channel, w, h)
        # input channel is channel from previous layer
        # while passing error backward, we calculate error cased by this layer's weights 
        # so transpose weight as : (input_channel, num_kernel, w, h)
        nweight = self.weights.copy()
        nweight = np.transpose(nweight, axes=(1,0,2,3)) if self.conv_dim==2 else np.transpose(nweight, axes=(1,0,2))
        ishape = self.input_tensor.shape
        y,x = ishape[-2:] if self.conv_dim==2 else (ishape[-1],None)

        bsize = self.input_tensor.shape[0]
        wk, wc = nweight.shape[:2]

        for cb in range(bsize):
            for ck in range(wk):
                grad = 0
                for c in range(wc):
                    if self.conv_dim==2:
                        err = np.zeros((y,x))
                        err[::sh, ::sw] = error_tensor[cb, c]
                    else:
                        err = np.zeros(y)
                        err[::sh] = error_tensor[cb, ck]
                    # we used correlate on forward, use convolve now
                    grad += signal.convolve(err, nweight[ck, c], mode='same', method='direct')
                    
                gradient[cb, ck] = grad
        return gradient

    def backward(self, error_tensor):
        self.update_parameters(error_tensor)
        gradient = self.error_this_layer(error_tensor)
        


        return gradient

    @property
    def gradient_weights(self):
        return self._gradient_weights

    @gradient_weights.setter
    def gradient_weights(self, value):
        self._gradient_weights = value

    @property
    def gradient_bias(self):
        return self._gradient_bias

    @gradient_bias.setter
    def gradient_bias(self, value):
        self._gradient_bias = value

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, value):
        self._optimizer = value

    @property
    def bias_optimizer(self):
        return self._bias_optimizer

    @bias_optimizer.setter
    def bias_optimizer(self, value):
        self._bias_optimizer = value
