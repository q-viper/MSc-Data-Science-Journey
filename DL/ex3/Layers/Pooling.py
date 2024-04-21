import numpy as np
from Layers.Base import BaseLayer


class Pooling(BaseLayer):

    def __init__(self, stride_shape, pooling_shape):
        super().__init__()
        self.stride_shape = stride_shape
        self.pooling_shape = pooling_shape
        
    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        bsize,channel,h,w = input_tensor.shape
        sh,sw=self.stride_shape
        ph,pw=self.pooling_shape
        
        # output shape will be ((w-f)/s+1), w is input width, f is kernel width, s is stride width
        hout = int(1 + (h - ph) / sh)
        wout = int(1 + (w - pw) / sw)

        output = np.zeros((bsize, channel, hout, wout))

        for cb in range(bsize):
            for c in range(channel):
                for ho in range(hout):
                    for wo in range(wout):
                        output[cb, c, ho, wo] = np.max(input_tensor[cb, c,ho * sh: ho * sh + ph,
                                                wo * sw: wo * sw + pw])
        self.output=output
        return output

    def backward(self, error_tensor):
        gradient = np.zeros_like(self.input_tensor)
        bsize,channel,h,w = self.input_tensor.shape
        sh,sw=self.stride_shape
        ph,pw=self.pooling_shape

        hout, wout = self.output.shape[-2:]

        # send error from previous layers only via those index which was selected during maxpooling!
        for cb in range(bsize):
            for ch in range(channel):
                for ho in range(hout):
                    for wo in range(wout):
                        # pass error through that index which was selected during maxpool
                        # find the index of value in a patch
                        xi, yi = np.where(self.output[cb,ch,ho,wo] == self.input_tensor[cb, ch, ho * sh: ho * sh + ph,
                                                                    wo * sw: wo * sw + pw])

                        xi, yi = xi[0], yi[0]

                        gradient[cb, ch, ho * sh: ho * sh + ph,wo * sw: wo * sw + pw][xi, yi] += error_tensor[cb, ch, ho, wo]

        return gradient