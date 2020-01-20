import torch
import torch.nn as nn
import numpy as np 
from torch.nn import functional as F
from models.commonBlocks import Linear, Conv2D, ModulatedConv2D, getActivation

class Decoder(nn.Module):
    """
    StyleGAN2 decoder
    """    
    def __init__(self, resolution = 64, dLatentSize = 256, fmapBase = 4096, fmapDecay = 1., fmapMax = 256, fmapMin = 1, activation = 'lrelu',
                scaleWeights = True, inCh = 3, downsample = 'bilinear', mode = 'resnet', **kwargs):
        super().__init__()
        self.resolution = resolution
        self.fmapBase = fmapBase
        self.fmapDecay = fmapDecay
        self.fmapMax = fmapMax
        self.fmapMin = fmapMin
        self.activation = getActivation(activation)
        self.scaleWeights = scaleWeights
        self.inCh = inCh
        self.downsample = downsample
        assert mode in ['skip','resnet'], f'Decoder ERROR: Invalid synthesis network architecture {mode}'
        self.mode = mode
        
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

        self.logits = Linear(inCh=nf(0),outCh=dLatentSize,scaleWeights=self.scaleWeights)

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

        t = self.logits(t)
        
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

        x = self.logits(x)

        return x