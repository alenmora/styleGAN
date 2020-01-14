import torch
import torch.nn as nn
import numpy as np 
from torch.nn import functional as F

def getActivation(name):
    if name == 'lrelu':
        return nn.LeakyReLU(negative_slope=0.2)
    if name == 'relu':
        return nn.LeakyReLU()
    if name == 'tanh':
        return nn.Tanh()
    if name == 'sigmoid':
        return nn.Sigmoid() 
    else:
        print('Activation function ERROR: The specified activation function is not a valid one')

class Linear(nn.Module):
    """
    Dense linear layer, with the option of weight scaling. If true, before the output, it
    equalizes the learning rate for the weights by scaling them using the normalization constant
    from He's initializer
    """
    def __init__(self, inCh, outCh, gain=np.sqrt(2), scaleWeights=True):
        super().__init__()
        self.linear = nn.Linear(inCh, outCh)
        
        # new bias to use after wscale
        self.bias = self.linear.bias
        self.linear.bias = None
        
        # calc wt scale
        initStd = 1.
        self.wtScale = gain/np.sqrt(inCh+outCh)
        
        if not scaleWeights:
            initStd = gain/np.sqrt(inCh+outCh)
            self.wtScale = 1.
        
        # init
        nn.init.normal_(self.linear.weight, mean=0.0, std=initStd)
        nn.init.constant_(self.bias, val=0)
        
        self.name = '(inp = %s)' % (self.linear.__class__.__name__ + str(self.linear.weight.shape))
        
    def forward(self, x):
        output = self.linear(x)*self.wtScale + self.bias.view(1, self.bias.shape[0])
        return output 

    def __repr__(self):
        return self.__class__.__name__ + self.name   

class Conv2D(nn.Module):
    """
    2D convolutional layer, with 'same' padding (output and input have the same size), and with the option of weight scaling. 
    If true, before the output, it equalizes the learning rate for the weights by scaling them using the normalization constant
    from He's initializer
    """
    def __init__(self, inCh, outCh, kernelSize, padding='same', gain=np.sqrt(2), scaleWeights=True):
        super().__init__()
        if padding == 'same': #Make sure the output tensors for each channel are the same size as the input ones
            assert stride = 1, 'Conv2D Module ERROR: padding option "same" is only supported for stride of 1'
            padding = kernelSize // 2
            
        self.conv = nn.Conv2d(in_channels=inCh, out_channels=outCh, kernel_size=kernelSize, stride=1, padding=padding)
        
        # new bias to use after wscale
        self.bias = self.conv.bias
        self.conv.bias = None
        
        # calc wt scale
        convShape = list(self.conv.weight.shape)
        fanIn = np.prod(convShape[1:]) # Leave out # of op filters
        initStd = 1.
        self.wtScale = gain/np.sqrt(fanIn)

        if not scaleWeights:
            initStd = gain/np.sqrt(fanIn)
            self.wtScale = 1.
        
        # init
        nn.init.normal_(self.conv.weight, mean=0.0, std=initStd)
        nn.init.constant_(self.bias, val=0)
        
        self.name = '(inp = %s)' % (self.conv.__class__.__name__ + str(convShape))
        
    def forward(self, x):
        output = self.conv(x)*self.wtScale + self.bias.view(1, self.bias.shape[0], 1, 1)
        return output 

    def __repr__(self):
        return self.__class__.__name__ + self.name

class ModulatedConv2D(nn.Module):
    """
    Modulated 2D convolutional layer. This is a 2D convolutional layer whose weights are modulated by an output of a linear
    network which maps the hidden latent vector to a style, and then demodulated (by scaling them) to a standad deviation of one. 
    It also has the option of weight scaling , which, if true, before the output, equalizes the learning rate for the original weights
    of the convolutional network and for the linear network used for modulation
    """
    def __init__(self, inStyle, inCh, outCh, kernelSize, padding='same', gain=np.sqrt(2), scaleWeights=True):
        super().__init__()
        assert kernelSize > 1 and kernelSize % 2 == 1, 'Conv2D Error: The kernel size must be an odd integer bigger than one' 
        if padding == 'same': #Make sure the output tensors for each channel are the same size as the input ones
            assert stride = 1, 'Conv2D Module ERROR: padding option "same" is only supported for stride of 1'
            padding = kernelSize // 2
        
        self.conv = nn.Conv2d(in_channels=inCh, out_channels=outCh, kernel_size=kernelSize, stride=1, padding=padding)
        
        # No conv bias
        self.conv.bias = None
        
        # Get weights
        self.weight = self.conv.weight
        
        # calc wt scale
        convShape = list(self.conv.weight.shape)
        fanIn = np.prod(convShape[1:]) # Leave out # of op filters
        initStd = 1.
        self.wtScale = gain/np.sqrt(fanIn)

        if not scaleWeights:
            initStd = gain/np.sqrt(fanIn)
            self.wtScale = 1.
        
        # init
        nn.init.normal_(self.weight, mean=0.0, std=initStd)

        #We need 1 scaling parameter per each input channel
        self.linear = Linear(inStyle, inCh, scaleWeights=scaleWeights)
        
        self.name = f'ModulatedConv2D: convolution {inCh} --> {outCh}; style length: {inStyle}'
        
    def forward(self, x, y):
        s = self.linear(y)
        self.weight.mul_(s.view(style.shape(0),1,style.shape(1),1,1))               #N x outCh x inCh x k x k - Modulate by multiplication over the inCh dimension
        norm = view(self.weight.shape(0),self.weight.shape(1),-1).norm(dim=3)+1e-8  #N x outCh - Norm for demodulation, which is calculated for each batch over the input weights of the same channel
        self.weight.div_(norm.view(*norm.shape,1,1))                                #N x outCh x inCh x k x k - Demodulate by dividing over the norm 
        print(self.conv.weight - self.weight)
        output = self.conv(x)*self.wtScale                                          #N x outCh x W x H

        return output 

    def __repr__(self):
        return self.__class__.__name__ + self.name

