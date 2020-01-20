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
    def __init__(self, inCh, outCh, gain=np.sqrt(2), bias=True, scaleWeights=True, lrmul = 1):
        super().__init__()
        
        # calc wt scale
        initStd = 1./lrmul
        self.wtScale = lrmul*gain/np.sqrt(inCh+outCh)

        self.lrmul = lrmul

        if not scaleWeights:
            initStd = gain/(lrmul*np.sqrt(inCh+outCh))
            self.wtScale = lrmul

        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(outCh))
        else:
            self.bias = None
        
        # init
        self.weight = torch.nn.Parameter(torch.zeros(outCh, inCh))

        nn.init.normal_(self.weight, mean=0.0, std=initStd)
        
        self.name = f'Linear module: {inCh} --> {outCh}'
        
    def forward(self, x):
        bias = None
        if self.bias is not None: bias = self.bias*self.lrmul
        return F.linear(x, self.weight*self.wtScale, bias)

    def __repr__(self):
        return self.name   

class Conv2D(nn.Module):
    """
    2D convolutional layer, with 'same' padding (output and input have the same size), and with the option of weight scaling. 
    If true, before the output, it equalizes the learning rate for the weights by scaling them using the normalization constant
    from He's initializer
    """
    def __init__(self, inCh, outCh, kernelSize, padding='same', gain=np.sqrt(2), scaleWeights=True, bias=True, lrmul = 1):
        super().__init__()
        if padding == 'same': #Make sure the output tensors for each channel are the same size as the input ones
            padding = kernelSize // 2

        self.padding = padding

        self.lrmul = lrmul

        # new bias to use after wscale
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(outCh))
        else:
            self.bias = None
        
        # calc wt scale
        fanIn = inCh*kernelSize*kernelSize # Leave out number of outCh
        initStd = 1./lrmul
        self.wtScale = lrmul*gain/np.sqrt(fanIn)

        if not scaleWeights:
            initStd = gain/(lrmul*np.sqrt(fanIn))
            self.wtScale = lrmul


        self.weight = nn.Parameter(torch.zeros(outCh,inCh,kernelSize,kernelSize))

        # init
        nn.init.normal_(self.weight, mean=0.0, std=initStd)
        
        self.name = 'Convolution2D Module '+ str(self.weight.shape)
        
    def forward(self, x):
        output = F.conv2d(x, 
                            self.wtScale*self.weight, 
                            padding = self.padding,
                            bias = self.bias*self.lrmul if self.bias is not None else None)
        
        return output 

    def __repr__(self):
        return self.name

class ModulatedConv2D(nn.Module):
    """
    Modulated 2D convolutional layer. This is a 2D convolutional layer whose weights are modulated by an output of a linear
    network which maps the hidden latent vector to a style, and then demodulated (by scaling them) to a standad deviation of one. 
    It also has the option of weight scaling , which, if true, before the output, equalizes the learning rate for the original weights
    of the convolutional network and for the linear network used for modulation
    """
    def __init__(self, inStyle, inCh, outCh, kernelSize, padding='same', gain=np.sqrt(2), bias=False, lrmul = 1, scaleWeights=True):
        super().__init__()
        assert kernelSize > 1 and kernelSize % 2 == 1, 'Conv2D Error: The kernel size must be an odd integer bigger than one' 
        if padding == 'same': #Make sure the output tensors for each channel are the same size as the input ones
            padding = kernelSize // 2
        
        self.kernelSize = kernelSize
        self.padding = padding

        self.lrmul = lrmul
        
        # Get weights
        self.weights = nn.Parameter(torch.zeros(1,outCh,inCh,self.kernelSize,self.kernelSize), requires_grad=True)
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(outCh), requires_grad=True)
        else:
            self.bias = None
        
        # calc wt scale
        fanIn = inCh*kernelSize*kernelSize # Leave out number of outCh
        initStd = 1./lrmul
        self.wtScale = lrmul*gain/np.sqrt(fanIn)

        if not scaleWeights:
            initStd = gain/(lrmul*np.sqrt(fanIn))
            self.wtScale = lrmul
        
        # init
        nn.init.normal_(self.weights, mean=0.0, std=initStd)

        #We need 1 scaling parameter per each input channel
        self.linear = Linear(inStyle, inCh, scaleWeights=scaleWeights)
        
        self.name = f'ModulatedConv2D: convolution {inCh} --> {outCh}; style length: {inStyle}'
        
    def forward(self, x, y):
        batchSize = x.size(0)
        s = self.linear(y)                                                                 #N x inCh
        s = s.view(batchSize,1,s.size(1),1,1)                                              #N x 1     x inCh x 1 x 1 
        modul = self.weights.mul(s)                                                        #N x outCh x inCh x k x k - Modulate by multiplication over the inCh dimension
        norm = modul.view(batchSize, modul.size(1), -1).norm(dim=2, keepdim=True)+1e-8     #N x outCh x 1 - Norm for demodulation, which is calculated for each batch over the input weights of the same channel
        demodul = modul.div(norm.view(*norm.shape,1,1))                                    #N x outCh x inCh x k x k - Demodulate by dividing over the norm 
        
        output = []
        for i in range(s.size(0)):
            bias = None
            if self.bias is not None: bias = self.bias*self.lrmul
            output.append(F.conv2d(x[i].view(1,*x[i].shape),
                                    demodul[i]*self.wtScale, 
                                    padding=self.padding, 
                                    bias = bias
                                  ))    #1 x outCh x H x W
        
        output = torch.cat(output, dim=0)  #N x outCh x H x W
        
        return output

    def __repr__(self):
        return self.name

