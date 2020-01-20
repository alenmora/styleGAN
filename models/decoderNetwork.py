import torch
import torch.nn as nn
import numpy as np 
from torch.nn import functional as F
from commonBlocks import Linear, Conv2D, ModulatedConv2D, getActivation

class Decoder(nn.Module):
    """
    Decoder for style transfer
    """    
    super().__init__()
    def __init__(self, config):
        self.nChannels = config.nChannels
        self.resolution = config.resolution
        self.fmapBase = config.fmapBase
        self.fmapDecay = config.fmapDecay
        self.fmapMax = config.fmapMax
        self.activation = getActivation(config.activationFunction)
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.scaleWeights = config.scaleWeights
        self.inCh = config.outputChannels
        self.downsample = config.upsampleMode
        assert config.decoderNetwork in ['revised','skip','resnet'], f'Decoder ERROR: Invalid synthesis network architecture {config.decoderNetwork}'
        self.mode = config.decoderNetwork
        self.latentSize = config.latentSize
        
        rlog2 = int(np.log2(self.resolution))
        assert self.resolution == 2**(relog2) and self.resolution >= 4, 'Decoder ERROR: The resolution should be a power of 2 greater than 4'

        def nf(stage): #Get the number of channels per layer
            return np.clip(int(self.fmapBase / (2.0 ** (stage * self.fmapDecay))), self.fmapMin, self.fmapMax)

        self.nLayers = 2*(rlog2-1)-1 #4x4 requires 1 (conv) layer, 8x8 requires 3, 16x16 requires 5,...

        self.convs = nn.ModuleList() #Keeps the 2D convolutional modules
        self.fromRGB = nn.ModuleList() #Keeps the toRGB modules
        self.lp = nn.ModuleList()    #Keeps the 2DConv modules for linear projection when performing resnet architecture

        def layer(kernel, layerId): #Constructor of layers
            resol = int(2**((layerId+5)//2)) #Recover the resolution of the current layer from its id (0 --> 4), (1 --> 8), (2 --> 8), (3 --> 16),...
            inCh = nf(layerId+1)
            outCh = nf(layerId)
            
            if not layerId % 2: #Even layer
                if self.mode != 'resnet': #add the fromRGB module for the given resolution
                    self.fromRGB.append(nn.Sequential(
                                            Conv2D(inCh=self.inCh, outCh=inCh, kernelSize=1, scaleWeights=self.scaleWeights),
                                            self.activation,
                                        ))
                
                else: #Add the convolution modules for properly matching the channels during the residual connection
                    if layerId > 0: # (the first layer does not require this module)
                        self.lp.append(Conv2D(inCh=inCh, outCh=nf(layerId-2)))

            #Add the required convolutional module
            if layerId == 0:
                self.convs.append(ModulatedConv2D(inStyle=self.dLatentSize, inCh=inCh, outCh=outCh, kernel=4, padding=0))
            else:
                self.convs.append(ModulatedConv2D(inStyle=self.dLatentSize, inCh=inCh, outCh=outCh, kernel=3))
        
        for layerId in range(self.nLayers): #Create the layers from to self.nLayers-1
            layer(kernel=3, layerId=maxLayer)  

        if self.mode == 'resnet': #Add the only toRGB module in the resnet architecture
            self.fromRGB.append(nn.Sequential(
                                    Conv2D(inCh=self.inCh, outCh=nf(self.nLayers), kernelSize=1, scaleWeights=self.scaleWeights),
                                    self.activation,
                                ))


        inCh = nf(0) if self.stdGroupSize <= 1 else nf(0)+1
        self.fullyConnected = Linear(inCh=nf(0),outCh=self.latentSize,scaleWeights=self.scaleWeights)

    def forward(self, x, *args, **kwargs):
        """
        Forward function.
        x (tensor): the input image
        *args, **kwargs: extra arguments for the forward step in the pogressive growing configuration
        """
        if self.mode == 'revised':
            maxLayer = kwargs['maxLayer']
            fadeWt = kwargs['fadeWt']
            return self.forwardProgressive_(x,maxLayer=maxLayer,fadeWt=fadeWt)
        elif self.mode == 'skip':
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
       
    def forwardProgressive_(self, x, maxLayer = None, fadeWt = 1.):
        """
        Perform a forward pass using
        the progressive growing architecture
        """
        if maxLayer == None:
            maxLayer = len(self.convs)-1

        assert maxLayer % 2 == 0, f'Synthesis Module ERROR: the layer ID in the forward call must be an even integer ({layerID})'
        assert maxLayer < len(self.convs), f'Synthesis Module ERROR: the layer ID {maxLayer} is out of bounds {len(self.convs)}'
        
        if fadeWt < 1:   #We are in a fade stage
            prev_x = F.interpolate(x, scale_factor=0.5, mode=self.downsample) #Downscale the image
            prev_x = self.toRGB[maxLayer//2-1](prev_x) #Transform it to RGB

            x = self.fromRGB[maxLayer//2](x) #Get the input formated from the current level
            x = self.applyOneLayer(x, maxLayer)    #Process the top level  

            for layer in range(maxLayer-1,-1,-1): #Apply the rest of the levels, from top to bottom
                x = self.applyOneLayer(x,layer)
                prev_x = self.applyOneLayer(prev_x,layer)
                if layer % 2: # Odd layer, must perform downsampling
                    x = F.interpolate(x,scale_factor=0.5,mode=self.downsample)
                    prev_x = F.interpolate(prev_x,scale_factor=0.5,mode=self.downsample)

            x = fadeWt*x + (1-fadeWt)*prev_x 

        else:
            x = self.fromRGB[maxLayer//2](x) #Get the input formated from the current level
            for layer in range(layer,-1,-1): #Apply the rest of the levels, from top to bottom
                x = self.applyOneLayer[layer](x)
                if layer % 2:
                    x = F.interpolate(x,scale_factor=0.5,mode=self.downsample)

        x = self.fullyConnected(x)
        
        return x
    
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
                t = F.interpolate(t, scale_factor=0.5, mode=self.downsample)
                x = F.interpolate(x, scale_factor=0.5, mode=self.downsample)

        t = self.fullyConnected(t)
        
        return t

    def forwardResnet_(self, x):
        """
        Perform a forward pass using
        the architecture with residual networks
        """"
        x = self.fromRGB[0](x) #Use the only fromRGB for this net
        carryover = 0
        for layer in range(self.nLayers-1,-1,-1): #Apply all layers
            if not layer % 2:  #Even layer
                if carryover:
                    x = (carryover + x)/np.sqrt(2)
                carryover = x
            
            x = self.applyOneLayer(x, layer)

            if layer % 2: #Odd layer, downsample
                x = F.interpolate(x, scale_factor=0.5, mode=self.downsample)
                carryover = self.lp[layer//2](carryover)
                carryover = F.interpolate(carryover, scale_factor=0.5, mode=self.downsample)

        x = self.fullyConnected(x)

        return x