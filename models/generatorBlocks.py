import torch
import torch.nn as nn
import numpy as np 
from torch.nn import functional as F
from models.commonBlocks import Linear, Conv2D, ModulatedConv2D, getActivation

class Mapping(nn.Module):
    """
    StyleGAN2 mapping generator module
    """    
    
    def __init__(self, latentSize=256, dLatentSize=256, mappingLayers = 4, neuronsInMappingLayers = 256, lrmul = 0.01, 
                activation = 'lrelu', scaleWeights = False, normalizeLatents = True, **kwargs):
        super().__init__()
        self.latentSize = latentSize
        self.dLatentSize = dLatentSize
        self.mappingLayers = mappingLayers
        assert self.mappingLayers > 0, 'Mapping Module ERROR: The number of mapping layers should be a positive integer' 
        self.normalizeLatents = normalizeLatents
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
            inCh = outCh

        self.map = nn.Sequential(*mods)

        self.name = 'Mapping subnetwork: '+str(self.map)
    
    def forward(self, x):
        if self.normalizeLatents:
            x*torch.rsqrt((x**2).mean(dim=1, keepdim=True)+1e-8)
        
        return self.map(x)

    def __repr__(self):
        return self.name

class NoiseLayer(nn.Module):
    """
    Module that adds the noise to the ModulatedConv2D output
    """
    def __init__(self, outCh, randomizeNoise = False):
        super().__init__()
        self.noise = None
        self.register_buffer('cached_noise', self.noise)
        self.randomizeNoise = randomizeNoise
        self.weights = nn.Parameter(torch.zeros(1,outCh,1,1), requires_grad=True)
        self.name = 'Noise layer: '+str(outCh)

    def forward(self, x):
        if self.randomizeNoise:
            noise = torch.randn(1,1,x.size(2),x.size(3), device=x.device)
            return x+self.weights*noise
        elif self.noise is None:
            self.noise = torch.randn(1,1,x.size(2),x.size(3), device=x.device)   
            self.register_buffer('cached_noise',self.noise)
            return x+self.weights*self.noise
        else:
            return x+self.weights*self.noise

    def __repr__(self):
        return self.name

class Synthesis(nn.Module):
    """
    StyleGAN2 original synthesis network
    """    
    def __init__(self, dLatentSize = 256, resolution = 64, fmapBase = 2048, fmapDecay = 1, fmapMax = 256, fmapMin = 1,
                randomizeNoise = False, activation = 'lrelu', scaleWeights = False, outCh = 3, upsample = 'bilinear', mode = 'skip', **kwargs):
        super().__init__()
        self.dLatentSize = dLatentSize
        self.resolution = resolution
        self.fmapBase = fmapBase
        self.fmapDecay = fmapDecay
        self.fmapMax = fmapMax
        self.fmapMin = fmapMin
        self.randomizeNoise = randomizeNoise
        self.activation = getActivation(activation)
        self.scaleWeights = scaleWeights
        self.outCh = outCh
        self.upsample = upsample
        self.mode = mode

        assert self.mode in ['skip','resnet'], f'Generator ERROR: Invalid synthesis network architecture {self.mode}'
        
        rlog2 = int(np.log2(self.resolution))
        assert self.resolution == 2**(rlog2) and self.resolution >= 4, f'Synthesis Module ERROR: The resolution should be a power of 2 greater than 4 ({self.resolution})'

        def nf(stage): #Get the number of channels per layer
            return np.clip(int(self.fmapBase / (2.0 ** (stage * self.fmapDecay))), self.fmapMin, self.fmapMax)

        self.nLayers = 2*rlog2-3 #a maximum resolution of 4x4 requires 1 layer, 8x8 requires 3, 16x16 requires 5,...

        self.convs = nn.ModuleList()        #Keeps the modulated convolutional modules
        self.noise = nn.ModuleList()        #Keeps the modules that perform noise broadcasting
        self.biases = nn.ParameterList()    #Keeps the biases added with the noise
        self.toRGB = nn.ModuleList()        #Keeps the toRGB modules
        self.lp = nn.ModuleList()           #Keeps the 2DConv modules for linear projection when performing resnet architecture

        self.cInput = nn.Parameter(torch.randn(1,nf(1),4,4)) #Constant random input
        self.register_parameter('cInput', self.cInput)

        def layer(kernel, layerId): #Constructor of layers
            resol = int(2**((layerId+5)//2)) #Recover the resolution of the current layer from its id (0 --> 4), (1 --> 8), (2 --> 8), (3 --> 16),...
            stage = int(np.log2(resol)-2) #Resolution stage: (4x4 --> 0), (8x8 --> 1), (16x16 --> 2) ...
            inCh = nf(stage) 
            outCh = nf(stage) if layerId % 2 else nf(stage+1) #The even layers give the output for the resolution block, so their number of outCh must be the same of the inCh for the next stage
            
            if not layerId % 2: #Even layer
                if self.mode != 'resnet': #add the toRGB module for the given resolution
                    self.toRGB.append(Conv2D(inCh=outCh, outCh=self.outCh, kernelSize=1, scaleWeights=self.scaleWeights))
                
                else: #Add the convolution modules for properly matching the channels during the residual connection
                    if layerId < self.nLayers-1: # (the last layer --which is even-- does not require this module)
                        self.lp.append(Conv2D(inCh=inCh, outCh=outCh,kernelSize=kernel))

            #Add the required modulated convolutional module
            self.convs.append(ModulatedConv2D(inStyle=self.dLatentSize, inCh=inCh, outCh=outCh, kernelSize=kernel))
            
            #Add the random noise broadcasting
            if self.randomizeNoise: 
                self.noise.append(NoiseLayer(outCh)) #Generate new noise every forward pass
            else:
                self.noise.append(NoiseLayer(outCh)) #Use the cached noise
        
            #Add the biases
            self.biases.append(nn.Parameter(torch.zeros(1,outCh,1,1), requires_grad=True)) 
        
        for layerId in range(self.nLayers): #Create the layers from to self.nLayers-1
            layer(kernel=3, layerId=layerId)  

        if self.mode == 'resnet': #Add the only toRGB module in the resnet architecture
            self.toRGB.append(Conv2D(inCh=nf((self.nLayers+1)//2),outCh=self.outCh, kernelSize=1, scaleWeights=self.scaleWeights))

    def forward(self, y, *args, x = None, **kwargs):
        """
        Forward function.
        y (tensor): the disentangled latent vector
        x (tentsor): the constant input map
        *args, **kwargs: extra arguments for the forward step in the pogressive growing configuration
        """
        bs = y.size(0)
        if x is None: x = self.cInput.repeat_interleave(bs,dim=0) #Repeat constant input for each style
        if self.mode == 'skip':
            return self.forwardSkip_(x,y)
        elif self.mode == 'resnet':
            return self.forwardResnet_(x,y)

    def forwardTo(self, y, maxLayer, x = None):
        """
        Forward tensor y up to layer maxLayer
        y (tensor): the disentangled latent vector
        maxLayer (int): the layer to forward the tensor up to
        x (tentsor): the constant input map
        """
        bs = y.size(0)
        if x is None: x = self.cInput.repeat_interleave(bs,dim=0) #Repeat constant input for each style
        assert maxLayer <= self.nLayers, f'Module Synthesis ERROR: The maxLayer {maxLayer} value in the forwardTo function is larger than the number of layers in the network {self.nLayers}'
        assert maxLayer >= 0, f'Module Synthesis ERROR: The maxLayer {maxLayer} value in the forwardTo function must be a nonnegative integer'
        if self.mode == 'skip':
            return self.forwardSkip_(x,y,maxLayer=maxLayer, getExtraOutputs=True)
        elif self.mode == 'resnet':
            return self.forwardResnet_(x,y,maxLayer=maxLayer, getExtraOutputs=True)

    def forwardFrom(self, y, x, extraInput, minLayer):
        """
        Forward tensor y up to layer maxLayer
        y (tensor): the disentangled latent vector
        x (tensor): the constant input map
        extraInput (tensor): for the skip and resnet configs, the carryover and output terms from the previous configuration
        minLayer(int): the layer from which to start the forwarding
        """
        assert minLayer <= self.nLayers, f'Module Synthesis ERROR: The maxLayer {minLayer} value in the forwardFrom function is larger than the number of layers in the network {self.nLayers}'
        assert minLayer >= 0, f'Module Synthesis ERROR: The maxLayer {minLayer} value in the forwardFrom function must be a nonnegative integer'
        if self.mode == 'skip':
            return self.forwardSkip_(x,y,output=extraInput,minLayer=minLayer)
        elif self.mode == 'resnet':
            return self.forwardResnet_(x,y,carryover=extraInput,minLayer=minLayer)

    def paTerm(self, y, x = None, againstInput = 1):
        """
        pull away term function to call
        y (tensor): the disentangled latent vector
        x (tentsor): the constant input map
        """
        bs = y.size(0)
        if x is None: x = self.cInput.repeat_interleave(bs,dim=0) #Repeat constant input for each style
        if  bs < 2: #Nothing to do if we only generate one candidate
            return 0
        
        fakes = self.forward(y, x = x)
        
        y = y.view(bs, -1) #Unroll
        fakes = fakes.view(bs, -1) #Unroll

        #Calculate pair-wise cosine similarities between batch elements 
        
        suma = 0
        for i in range(bs):
            for j in range(i+1,bs):
                fakesim = torch.nn.functional.cosine_similarity(fakes[i],fakes[j],dim=0)
                if againstInput == 0:
                    suma = suma + fakesim**2
                elif againstInput == 1:
                    ysim = torch.nn.functional.cosine_similarity(y[i],y[j],dim=0)
                    suma = suma + (ysim-fakesim)**2/(ysim**2 +1e-8)
                elif againstInput == 2:
                    ysim = torch.nn.functional.cosine_similarity(y[i],y[j],dim=0)
                    suma = suma + (ysim**2-fakesim**2)**2/(ysim**4 +1e-8)
                else:
                    return 0

        return suma/(bs*(bs-1))

    def applyOneLayer(self, x, y, layer):
        """
        Apply one layer of the generator to
        the constant tensor c and the latent 
        vector y
        """
        x = self.convs[layer](x,y)
        x = self.noise[layer](x)
        bias = self.biases[layer]
        x = x + bias
        x = self.activation(x)
        return x
           
    def forwardSkip_(self, x, y, minLayer = 0, maxLayer = None, output = 0, getExtraOutputs = False):
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

            x = self.applyOneLayer(x, y, layer)

            if not layer % 2:  #Even layer, so get the generated output for the given resolution, resize it, and add it to the final output
                output = output + self.toRGB[layer//2](x)
        
        if getExtraOutputs:
            return x, output 

        return output

    def forwardResnet_(self, x, y, minLayer = 0, maxLayer = None, carryover = None, getExtraOutputs = False):
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

            x = self.applyOneLayer(x, y, layer)

            if not layer % 2:  #Even layer, so add and actualize carryover value
                if carryover is not None: #If there is a carryover, add it to the output
                    x = (carryover + x)/np.sqrt(2)
                carryover = x

        x = self.toRGB[0](x) #Use the only toRGB for this net

        if getExtraOutputs:
            return x, carryover

        return x