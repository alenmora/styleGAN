import torch
import torch.nn as nn
import numpy as np 
from torch.nn import functional as F
from models.commonBlocks import Linear, Conv2D, ModulatedConv2D, getActivation

class MiniBatchStdDevLayer(nn.Module):
    """
    Add std to last layer group of critic to improve variance
    """
    def __init__(self, groupSize = 4):
        super().__init__()
        self.groupSize = groupSize

    def forward(self, x):
        shape = list(x.size())                                              # NCHW - Initial size
        xStd = x.view(self.groupSize, -1, shape[1], shape[2], shape[3])     # GMCHW - split minbatch into M groups of size G (= groupSize)
        xStd -= torch.mean(xStd, dim=0, keepdim=True)                       # GMCHW - Subract mean over groups
        xStd = torch.mean(xStd ** 2, dim=0, keepdim=False)                  # MCHW - Calculate variance over groups
        xStd = (xStd + 1e-08) ** 0.5                                        # MCHW - Calculate std dev over groups
        xStd = torch.mean(xStd.view(xStd.shape[0], -1), dim=1, keepdim=True).view(-1, 1, 1, 1)
                                                                            # M111 - Take mean over CHW
        xStd = xStd.repeat(self.groupSize, 1, shape[2], shape[3])           # N1HW - Expand to same shape as x with one channel 
        output = torch.cat([x, xStd], 1)
        return output
    
    def __repr__(self):
        return self.__class__.__name__ + '(Group Size = %s)' % (self.groupSize)

class Critic(nn.Module):
    """
    StyleGAN2 critics
    """    
    def __init__(self, resolution = 64, fmapBase = 4096, fmapDecay = 1., fmapMax = 256, fmapMin = 1, activation = 'lrelu',
                scaleWeights = True, inCh = 3, stdGroupSize = 8, downsample = 'bilinear', mode = 'resnet', asRanker = False, **kwargs):
        super().__init__()
        self.resolution = resolution
        self.fmapBase = fmapBase
        self.fmapDecay = fmapDecay
        self.fmapMax = fmapMax
        self.fmapMin = fmapMin
        self.activation = getActivation(activation)
        self.scaleWeights = scaleWeights
        self.inCh = inCh
        self.stdGroupSize = stdGroupSize
        self.downsample = downsample
        assert mode in ['skip','resnet'], f'Critic ERROR: Invalid synthesis network architecture {mode}'
        self.mode = mode
        self.asRanker = asRanker
        
        rlog2 = int(np.log2(self.resolution))
        assert self.resolution == 2**(rlog2) and self.resolution >= 4, 'Critic ERROR: The resolution should be a power of 2 greater than 4'

        def nf(stage): #Get the number of channels per layer
            return np.clip(int(self.fmapBase / (2.0 ** (stage * self.fmapDecay))), self.fmapMin, self.fmapMax)

        self.nLayers = 2*(rlog2-1)-1 #4x4 requires 1 (conv) layer, 8x8 requires 3, 16x16 requires 5,...

        self.convs = nn.ModuleList() #Keeps the 2D convolutional modules
        self.fromRGB = nn.ModuleList() #Keeps the toRGB modules
        self.lp = nn.ModuleList()    #Keeps the 2DConv modules for linear projection when performing resnet architecture

        def layer(kernel, layerId): #Constructor of layers
            stage = int((layerId+1)//2) #Resolution stage: (4x4 --> 0), (8x8 --> 1), (16x16 --> 2) ...
            inCh = nf(stage) if layerId % 2 else nf(stage+1) #The even layers receive the input of the resolution block, so their number of inCh must be the same of the outCh for the previous stage
            outCh = nf(stage)

            if not layerId % 2: #Even layer
                if self.mode != 'resnet': #add the fromRGB module for the given resolution
                    self.fromRGB.append(nn.Sequential(
                                            Conv2D(inCh=self.inCh, outCh=inCh, kernelSize=1, scaleWeights=self.scaleWeights),
                                            self.activation,
                                        ))
                
                else: #Add the convolution modules for properly matching the channels during the residual connection
                    if layerId > 0: # (the first layer does not require this module)
                        self.lp.append(Conv2D(inCh=inCh, outCh=outCh,kernelSize=kernel))

            #Add the required convolutional module
            if layerId == 0:
                self.convs.append(Conv2D(inCh=inCh, outCh=outCh, kernelSize=4, padding=0))
            else:
                self.convs.append(Conv2D(inCh=inCh, outCh=outCh, kernelSize=kernel))
        
        for layerId in range(self.nLayers): #Create the layers from to self.nLayers-1
            layer(kernel=3, layerId=layerId)  

        if self.mode == 'resnet': #Add the only toRGB module in the resnet architecture
            self.fromRGB.append(nn.Sequential(
                                    Conv2D(inCh=self.inCh, outCh=nf((self.nLayers+1)//2), kernelSize=1, scaleWeights=self.scaleWeights),
                                    self.activation,
                                ))

        if self.stdGroupSize > 1:
            self.miniBatchLayer = MiniBatchStdDevLayer(self.stdGroupSize)

        inCh = nf(0) if self.stdGroupSize <= 1 else nf(0)+1
        self.fullyConnected = Linear(inCh=inCh,outCh=1,scaleWeights=self.scaleWeights)

    def forward(self, x):
        """
        Forward function.
        x (tentsor): the input
        *args, **kwargs: extra arguments for the forward step in the pogressive growing configuration
        """
        if self.mode == 'skip':
            return self.forwardSkip_(x)
        elif self.mode == 'resnet':
            return self.forwardResnet_(x)
    
    def applyOneLayer(self, x, layer):
        """
        Apply one layer of the critic to
        the tensor x
        """
        x = self.convs[layer](x)
        return self.activation(x)

    def applyLastLayer(self, x):
        if self.stdGroupSize > 1 and not self.asRanker:
            x = self.miniBatchLayer(x)
        x = x.view(x.size(0),-1)            #Get rid of trivial dimensions
        return self.fullyConnected(x)
           
    def forwardSkip_(self, x):
        """
        Perform a forward pass using
        the architecture with skip connections
        """
        t = 0
        for layer in range(self.nLayers-1,-1,-1):
            if not layer % 2: #Even layer: get the fromRGB version of the downsampled image
                t = self.fromRGB[layer//2](x)+t
                
            t = self.applyOneLayer(t, layer)

            if layer % 2: #Downsample
                t = F.interpolate(t, scale_factor=0.5, mode=self.downsample, align_corners=False)
                x = F.interpolate(x, scale_factor=0.5, mode=self.downsample, align_corners=False)

        t = self.applyLastLayer(t)
        
        return t

    def forwardResnet_(self, x):
        """
        Perform a forward pass using
        the architecture with residual networks
        """
        x = self.fromRGB[0](x) #Use the only fromRGB for this net
        carryover = None
        for layer in range(self.nLayers-1,-1,-1): #Apply all layers
            if not layer % 2:  #Even layer
                if carryover is not None:
                    x = (carryover + x)/np.sqrt(2)
                carryover = x
            
            x = self.applyOneLayer(x, layer)

            if layer % 2: #Odd layer, downsample
                x = F.interpolate(x, scale_factor=0.5, mode=self.downsample, align_corners=False)
                carryover = self.lp[layer//2](carryover)
                carryover = F.interpolate(carryover, scale_factor=0.5, mode=self.downsample, align_corners=False)

        x = self.applyLastLayer(x)

        return x