import torch
import torch.nn as nn
import numpy as np 
from torch.nn import functional as F
from models.commonBlocks import PixelNorm, Linear, Conv2D, ModulatedConv2D, getActivation

class constantInput(nn.Module):
    def __init__(self, nCh, resol=4, makeTrainable = True):
        super().__init__()

        self.cInput = nn.Parameter(torch.randn(1,nCh,4,4)) #Constant random input
        self.cInput.requires_grad_(makeTrainable)

    def forward(self, input):
        batchSize = input.size(0)

        return self.cInput.repeat(batchSize, 1, 1, 1)

class Mapping(nn.Module):
    """
    StyleGAN2 mapping generator module
    """    
    
    def __init__(self, latentSize=256, dLatentSize=256, mappingLayers = 4, neuronsInMappingLayers = 256, lrmul = 0.01, 
                activation = 'lrelu', scaleWeights = False, normalizeLayers = False, **kwargs):
        super().__init__()
        self.latentSize = latentSize
        self.dLatentSize = dLatentSize
        self.mappingLayers = mappingLayers
        assert self.mappingLayers > 0, 'Mapping Module ERROR: The number of mapping layers should be a positive integer' 
        self.scaleWeights = scaleWeights
        self.nNeurons = neuronsInMappingLayers
        self.activation = getActivation(activation)
        self.lrmul = lrmul

        mods = []

        inCh = self.latentSize
        for layerId in range(self.mappingLayers):
            outCh = self.nNeurons if layerId != (self.mappingLayers-1) else self.dLatentSize
            mods.append(Linear(inCh, outCh, scaleWeights=self.scaleWeights, lrmul=self.lrmul))
            mods.append(self.activation)
            if normalizeLayers: mods.append(PixelNorm())
            inCh = outCh

        self.map = nn.Sequential(*mods)

        self.name = 'Mapping subnetwork: '+str(self.map)
    
    def forward(self, x):
        return self.map(x)

    def __repr__(self):
        return self.name

class NoiseLayer(nn.Module):
    """
    Module that adds the noise to the ModulatedConv2D output
    """
    def __init__(self, outCh, resolution, randomizeNoise = False):
        super().__init__()
        self.noise = torch.randn(1,1,resolution,resolution)   
        self.register_buffer('cached_noise', self.noise)
        self.randomizeNoise = randomizeNoise
        self.weights = nn.Parameter(torch.zeros(1,outCh,1,1), requires_grad=True)
        self.name = 'Noise layer: '+str(outCh)

    def forward(self, x): 
        noise = torch.randn(1,1,x.size(2),x.size(3), device=x.device) if self.randomizeNoise else self.noise.to(x.device)
        return x+self.weights*noise
        
    def __repr__(self):
        return self.name

class StyledConv2D(nn.Module):
    """
    Module representing the mixing of a modulated 2DConv and noise addition
    """
    def __init__(self, styleCh, inCh, outCh, kernelSize, resolution, padding='same', gain=np.sqrt(2), bias=False, lrmul = 1, scaleWeights=True, 
    demodulate = True, randomizeNoise = False, activation = 'lrelu'):
        super().__init__()

        self.conv = ModulatedConv2D(styleCh, inCh, outCh, kernelSize, padding=padding, gain=gain, bias=bias, lrmul=lrmul, scaleWeights=scaleWeights, demodulate=demodulate)
        self.noise = NoiseLayer(outCh, resolution, randomizeNoise=randomizeNoise)
        self.activation = getActivation(activation)
    
    def forward(self, x, y):
        out = self.conv(x, y)
        out = self.noise(out)
        out = self.activation(out)

        return out

    def __repr__(self):
        return 'StyledConv2D based on '+self.conv.__repr__()

class ToRGB(nn.Module):
    """
    Module to transform to image space
    """
    def __init__(self, styleCh, inCh, outCh):
        super().__init__()

        self.conv = ModulatedConv2D(styleCh, inCh, outCh, kernelSize = 1, demodulate=False)
        self.bias = nn.Parameter(torch.zeros(1, outCh, 1, 1))

    def forward(self, x, y):
        out = self.conv(x,y)
        out = out + self.bias

        return out

    def __repr__(self):
        return f'ToRGB using '+self.conv.__repr__()

class Synthesis(nn.Module):
    """
    StyleGAN2 original synthesis network
    """    
    def __init__(self, dLatentSize = 256, resolution = 64, fmapBase = 2048, fmapDecay = 1, fmapMax = 256, fmapMin = 1,
                randomizeNoise = False, activation = 'lrelu', scaleWeights = False, outCh = 3, upsample = 'bilinear', mode = 'skip', 
                normalizeLayers = False,**kwargs):
        super().__init__()
        self.dLatentSize = dLatentSize
        self.resolution = resolution
        self.fmapBase = fmapBase
        self.fmapDecay = fmapDecay
        self.fmapMax = fmapMax
        self.fmapMin = fmapMin
        self.activation = activation
        self.upsample = upsample
        self.mode = mode
        self.outCh = outCh
        self.normalizeLayers = normalizeLayers

        assert self.mode in ['skip','resnet'], f'Generator ERROR: Invalid synthesis network architecture {self.mode}'
        
        rlog2 = int(np.log2(self.resolution))
        assert self.resolution == 2**(rlog2) and self.resolution >= 4, f'Synthesis Module ERROR: The resolution should be a power of 2 greater than 4 ({self.resolution})'

        def nf(stage): #Get the number of channels per layer
            return np.clip(int(self.fmapBase / (2.0 ** (stage * self.fmapDecay))), self.fmapMin, self.fmapMax)

        self.nLayers = 2*rlog2-3 #a maximum resolution of 4x4 requires 1 layer, 8x8 requires 3, 16x16 requires 5,...

        self.styleConvs = nn.ModuleList()   #Keeps the style convolutional modules
        self.toRGB = nn.ModuleList()        #Keeps the ToRGB modules
        self.lp = nn.ModuleList()           #Keeps the 2DConv modules for linear projection when performing resnet architecture
        
        if self.normalizeLayers: self.normalizer = PixelNorm()       #Pixel normalizer

        def layer(kernel, layerId): #Constructor of layers
            resol = int(2**((layerId+5)//2)) #Recover the resolution of the current layer from its id (0 --> 4), (1 --> 8), (2 --> 8), (3 --> 16),...
            stage = int(np.log2(resol)-2) #Resolution stage: (4x4 --> 0), (8x8 --> 1), (16x16 --> 2) ...
            inCh = nf(stage) 
            outCh = nf(stage) if layerId % 2 else nf(stage+1) #The even layers give the output for the resolution block, so their number of outCh must be the same of the inCh for the next stage
            
            if not layerId % 2: #Even layer
                if self.mode == 'skip': #add the ToRGB module for the given resolution
                    self.toRGB.append(ToRGB(styleCh=self.dLatentSize, inCh=outCh, outCh=self.outCh))
                
                elif self.mode == 'resnet': #Add the convolution modules for properly matching the channels during the residual connection
                    if layerId < self.nLayers-1: # (the last layer --which is even-- does not require this module)
                        self.lp.append(Conv2D(inCh=inCh, outCh=outCh, kernelSize=1))

            #Add the required modulated convolutional module
            self.styleConvs.append(StyledConv2D(styleCh=self.dLatentSize, inCh=inCh, outCh=outCh, kernelSize=kernel, resolution=resol, randomizeNoise=randomizeNoise, activation=activation))
            
        for layerId in range(self.nLayers): #Create the layers from to self.nLayers-1
            layer(kernel=3, layerId=layerId)  

        if self.mode == 'resnet': #Add the only toRGB module in the resnet architecture
            self.toRGB.append(Conv2D(inCh=nf((self.nLayers+1)//2),outCh=self.outCh, kernelSize=1, scaleWeights=self.scaleWeights))

    def forward(self, x, w):
        """
        Forward function.
        y (tensor): the disentangled latent vector
        x (tentsor): the constant input map
        *args, **kwargs: extra arguments for the forward step in the pogressive growing configuration
        """
        if self.mode == 'skip':
            return self.forwardSkip_(x,w)
        elif self.mode == 'resnet':
            return self.forwardResnet_(x,w)

    def forwardTo(self, x, w, maxLayer):
        """
        Forward tensor y up to layer maxLayer
        y (tensor): the disentangled latent vector
        maxLayer (int): the layer to forward the tensor up to
        x (tentsor): the constant input map
        """
        assert maxLayer <= self.nLayers, f'Module Synthesis ERROR: The maxLayer {maxLayer} value in the forwardTo function is larger than the number of layers in the network {self.nLayers}'
        assert maxLayer >= 0, f'Module Synthesis ERROR: The maxLayer {maxLayer} value in the forwardTo function must be a nonnegative integer'
        if self.mode == 'skip':
            return self.forwardSkip_(x,w,maxLayer=maxLayer, getExtraOutputs=True)
        elif self.mode == 'resnet':
            return self.forwardResnet_(x,w,maxLayer=maxLayer, getExtraOutputs=True)

    def forwardFrom(self, x, w, extraInput, minLayer):
        """
        Forward tensor y up to layer maxLayer
        y (tensor): the disentangled latent vector
        x (tensor): the constant input map
        extraInput (tensor): for the skip and resnet configs, the carryover and output terms from the previous configuration
        minLayer(int): the layer from which to start the forwarding
        """
        assert minLayer <= self.nLayers, f'Module Synthesis ERROR: The minLayer {minLayer} value in the forwardFrom function is larger than the number of layers in the network {self.nLayers}'
        assert minLayer >= 0, f'Module Synthesis ERROR: The minLayer {minLayer} value in the forwardFrom function must be a nonnegative integer'
        if self.mode == 'skip':
            return self.forwardSkip_(x,w,output=extraInput,minLayer=minLayer)
        elif self.mode == 'resnet':
            return self.forwardResnet_(x,w,carryover=extraInput,minLayer=minLayer)

    def forwardSkip_(self, x, w, minLayer = 0, maxLayer = None, output = 0, getExtraOutputs = False):
        """
        Perform a forward pass using
        the architecture with skip connections
        """
        if maxLayer is None:
            maxLayer = self.nLayers

        for layer in range(minLayer, maxLayer): #Apply all layers
            if layer % 2: #Odd layer, so increase size
                x = F.interpolate(x, scale_factor=2, mode=self.upsample, align_corners=False)
                output = F.interpolate(output, scale_factor=2, mode=self.upsample, align_corners=False)

            x = self.styleConvs[layer](x, w)
            if self.normalizeLayers:
                x = self.normalizer(x)

            if not layer % 2:  #Even layer, so get the generated output for the given resolution, resize it, and add it to the final output
                output = output + self.toRGB[layer//2](x, w)
        
        if getExtraOutputs:
            return x, output 

        return output

    def forwardResnet_(self, x, w, minLayer = 0, maxLayer = None, carryover = None, getExtraOutputs = False):
        """
        Perform a forward pass using
        the architecture with residual networks
        """
        if maxLayer is None:
            maxLayer = self.nLayers

        for layer in range(minLayer, maxLayer): #Apply all layers
            if layer % 2: #Odd layer, so increase size
                x = F.interpolate(x, scale_factor=2, mode=self.upsample, align_corners=False)
                carryover = self.lp[layer//2](carryover)
                carryover = F.interpolate(carryover, scale_factor=2, mode=self.upsample, align_corners=False)

            x = self.styleConvs[layer](x, w)
            if self.normalizeLayers:
                x = self.normalizer(x)

            if not layer % 2:  #Even layer, so add and actualize carryover value
                if carryover is not None: #If there is a carryover, add it to the output
                    x = (carryover + x)/np.sqrt(2)
                carryover = x

        x = self.toRGB[0](x, w) #Use the only toRGB for this net

        if getExtraOutputs:
            return x, carryover

        return x