import torch
import torch.nn as nn
import numpy as np 
from torch.nn import functional as F
from commonBlocks import Linear, Conv2D, ModulatedConv2D, getActivation

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
    super().__init__()
    def __init__(self, config, asRanker = False):
        self.nChannels = config.nChannels
        self.resolution = config.resolution
        self.fmapBase = config.fmapBase
        self.fmapDecay = config.fmapDecay
        self.fmapMax = config.fmapMax
        self.activation = getActivation(config.activationFunction)
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.scaleWeights = config.scaleWeights
        self.inCh = config.outputChannels
        self.stdGroupSize = config.stdDevGroup
        self.downsample = config.upsampleMode
        assert config.criticNetwork in ['revised','skip','resnet'], f'Generator ERROR: Invalid synthesis network architecture {config.criticNetwork}'
        self.mode = config.criticNetwork
        self.asRanker = asRanker
        
        rlog2 = int(np.log2(self.resolution))
        assert self.resolution == 2**(relog2) and self.resolution >= 4, 'Synthesis Module ERROR: The resolution should be a power of 2 greater than 4'

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
            layer(kernel=3, layerId=layerId)  

        if self.mode == 'resnet': #Add the only toRGB module in the resnet architecture
            self.fromRGB.append(nn.Sequential(
                                    Conv2D(inCh=self.inCh, outCh=nf(self.nLayers), kernelSize=1, scaleWeights=self.scaleWeights),
                                    self.activation,
                                ))

        if self.stdGroupSize > 1:
            self.miniBatchLayer = MiniBatchStdDevLayer(self.stdGroupSize)

        inCh = nf(0) if self.stdGroupSize <= 1 else nf(0)+1
        self.fullyConnected = Linear(inCh=nf(0),outCh=1,scaleWeights=self.scaleWeights)

    def forward(self, x, *args, **kwargs):
        """
        Forward function.
        x (tentsor): the input
        *args, **kwargs: extra arguments for the forward step in the pogressive growing configuration
        """
        if self.mode == 'revised':
            return self.forwardProgressive_(x,*args,**kwargs)
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

    def applyLastLayer(self, x):
        if self.stdGroupSize > 1 and not self.asRanker:
            x = self.miniBatchLayer(x)

        return self.fullyConnected(x)
           
    def forwardProgressive_(self, x, layerId = None, fadeWt = 1.):
        """
        Perform a forward pass using
        the progressive growing architecture
        """
        assert layerId % 2 == 0, f'Synthesis Module ERROR: the layer ID in the forward call must be an even integer ({layerID})'
        asser layerId < len(self.convs), f'Synthesis Module ERROR: the layer ID {layerId} is out of bounds {len(self.convs)}'
        if layerId == None:
            layerId = len(self.convs)-1

        if fadeWt < 1:   #We are in a fade stage
            prev_x = F.interpolate(x, scale_factor=0.5, mode=self.downsample) #Downscale the image
            prev_x = self.toRGB[layerId//2-1](prev_x) #Transform it to RGB

            x = self.fromRGB[layerId//2](x) #Get the input formated from the current level
            x = self.applyOneLayer(x, layerId)    #Process the top level  

            for layer in range(layerId-1,-1,-1): #Apply the rest of the levels, from top to bottom
                x = self.applyOneLayer(x,layer)
                prev_x = self.applyOneLayer(prev_x,layer)
                if layer % 2: # Odd layer, must perform downsampling
                    x = F.interpolate(x,scale_factor=0.5,mode=self.downsample)
                    prev_x = F.interpolate(prev_x,scale_factor=0.5,mode=self.downsample)

            if self.stdGroupSize > 1 and not self.asRanker:
                x = self.miniBatchLayer(x)
                prev_x = self.miniBatchLayer(x)

            x = fadeWt*x + (1-fadeWt)*prev_x 

        else:
            x = self.fromRGB[layerId//2](x) #Get the input formated from the current level
            for layer in range(layer,-1,-1): #Apply the rest of the levels, from top to bottom
                x = self.applyOneLayer[layer](x)
                if layer % 2:
                    x = F.interpolate(x,scale_factor=0.5,mode=self.downsample)

            if self.stdGroupSize > 1 and not self.asRanker:
                x = self.miniBatchLayer(x)

        x = self.fullyConnected(x)
        
        return x
    
    def forwardSkip_(self, x):
        """
        Perform a forward pass using
        the architecture with skip connections
        """"
        t = 0
        for layer in range(self.nLayers-1,-1,-1):
            if not layer % 2: #Even layer: get the fromRGB version of the downsampled image
                t = self.fromRGB[layer//2](x)+t
                
            t = self.applyOneLayer(t, layer)

            if layer % 2: #Downsample
                t = F.interpolate(t, scale_factor=0.5, mode=self.downsample)
                x = F.interpolate(x, scale_factor=0.5, mode=self.downsample)

        t = self.applyLastLayer(t)
        
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

        x = self.applyLastLayer(x)

        return x

    def getGradientsWrtInputs(self, imp, *args, **kwargs):
        """
        Return the unrolled gradient matrix of the critic output wrt the input parameters for 
        each example in the input
        (should have the size batchSize x (imageChannels x imageWidth x imageHeight))
        """
        x = imp.detach().requires_grad_()
        out = self.forward(x, curResLevel=curResLevel, fadeWt=fadeWt)
        ddx = autograd.grad(outputs=out, inputs=x,
                              grad_outputs = torch.ones(out.size(),device=device),
                              create_graph = True, retain_graph=True, only_inputs=True)[0]
        ddx = ddx.view(ddx.size(0), -1)

        return ddx
