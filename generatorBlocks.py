import torch
import torch.nn as nn
import numpy as np 
from torch.nn import functional as F
from commonBlocks import Linear, Conv2D, ModulatedConv2D, getActivation

class Mapping(nn.Module):
    """
    StyleGAN2 mapping generator module
    """    
   def __init__(self, config):
        self.latentSize = config.latentSize
        self.dLatentSize = config.dLatentSize
        self.mappingLayers = config.mappingLayers
        assert self.mappingLayers > 0, 'Mapping Module ERROR: The number of mapping layers should be a positive integer' 
        self.normalizeLatents = config.normalizeLatents
        self.scaleWeights = config.scaleWeights
        self.nNeurons = config.neuronsInMappingLayers

        mods = []

        inCh = self.latentSize
        for layerId in range(self.mappingLayers):
            outCh = self.nNeurons if layerId != (self.mappingLayers-1) else self.dLatentSize
            mods.append(Linear(inCh, outCh, scaleWeights=self.scaleWeights))
            inCh = outCh

        self.map = nn.Sequential(*mods)

        self.name = 'Mapping subnetwork: '+str(self.map)
    
    def forward(self, x):
        if self.normalizeLatents:
            x.div_(x.norm(dim=1, keepdim=True))
        
        return self.map(x)

    def __repr__(self):
        return self.name

class addNoise(nn.Module):
    """
    Module that adds the noise to the ModulatedConv2D output
    """
    def __init__(self, outCh, noise = None, size = None, gain=np.sqrt(2), scaleWeights=True):
        assert noise != None or size != None, 'addNoise Module ERROR: Either the input noise or the noise size is needed'
        self.noise = noise.requires_grad_(False)
        self.size = size if size else noise.numel()
        fanIn = self.size 
        initStd = 1.
        self.wtScale = gain/np.sqrt(fanIn)
        if not scaleWeights:
            initStd = gain/np.sqrt(fanIn)
            self.wtScale = 1.
        
        # init
        self.weights = torch.zeros(1,outCh).requires_grad_(True)
        nn.init.normal_(self.weights, mean=0.0, std=initStd)

    def forward(self,x):
        if self.noise == None:
            noise = torch.randn(1,1,resol,resol, device=self.device).requires_grad_(False)
            return x+self.weights.view(1,outCh,1,1)*noise

        else:
            return x+self.weights.view(1,outCh,1,1)*self.noise

class Synthesis(nn.Module):
    """
    StyleGAN2 original synthesis network
    """    
    super().__init__()
    def __init__(self, config):
        self.dLatentSize = config.dLatentSize
        self.nChannels = config.nChannels
        self.resolution = config.resolution
        self.fmapBase = config.fmapBase
        self.fmapDecay = config.fmapDecay
        self.fmapMax = config.fmapMax
        self.randomizeNoise = config.randomizeNoise
        self.activation = getActivation(config.activationFunction)
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.scaleWeights = config.scaleWeights
        self.outCh = config.outputChannels
        self.upsample = config.upsampleMode
        self.mode = config.synthesisNetwork
        
        rlog2 = int(np.log2(self.resolution))
        assert self.resolution == 2**(relog2) and self.resolution >= 4, 'Synthesis Module ERROR: The resolution should be a power of 2 greater than 4'

        def nf(stage): #Get the number of channels per layer
            return np.clip(int(self.fmapBase / (2.0 ** (stage * self.fmapDecay))), self.fmapMin, self.fmapMax)

        self.nLayers = 2*(rlog2-1)-1 #4x4 requires 1 (conv) layer, 8x8 requires 3, 16x16 requires 5,...

        self.noiseInputs = [] #Store the noise inputs for each layer

        if not self.randomizeNoise: #Only if we are not randomizing it every iteration
            for layerId in range(self.nLayers):
                resol = 2**((layerId+5)//2)
                self.noiseInputs.append(torch.randn(1,1,resol,resol, device=self.device).requires_grad_(False))

        self.convs = nn.ModuleList() #Keeps the modulated convolutional modules
        self.noise = nn.ModuleList() #Keeps the modules that perform noise broadcasting
        self.biases = []             #Keeps the biases added with the noise
        self.toRGB = nn.ModuleList() #Keeps the toRGB modules
        self.lp = nn.ModuleList()    #Keeps the 2DConv modules for linear projection when performing resnet architecture

        self.cInput = torch.randn(1,nf(1),4,4, device=self.device).requires_grad_(False) #Constant random input

        def layer(kernel, layerId): #Constructor of layers
            resol = int(2**((layerId+5)//2)) #Recover the resolution of the current layer from its id (0 --> 4), (1 --> 8), (2 --> 8), (3 --> 16),...
            inCh = nf(layerId+1) if layerId % 2 else nf(layerId) #Keep the same number of channels for the whole block
            outCh = nf(layerId+1) if layerId % 2 else nf(layerId) #Keep the same number of channels for the whole block
            
            if not layerId % 2: #Even layer
                if self.mode != 'resnet': #add the toRGB module for the given resolution
                    self.toRGB.append(Conv2D(inCh=outCh, outCh=self.outCh, kernelSize=1, scaleWeights=self.scaleWeights))
                
                else: #Add the convolution modules for properly matching the channels during the residual connection
                    if layerId < self.nLayers-1: # (the last layer --which is even-- does not require this module)
                        self.lp.append(Conv2D(inCh=outCh, outCh=nf(layerId+1)))

            #Add the required modulated convolutional module
            self.convs.append(ModulatedConv2D(inStyle=self.dLatentSize, inCh=inCh, outCh=outCh, kernel=kernel))
            
            #Add the random noise broadcasting
            if self.randomizeNoise: 
                self.noise.append(addNoise(outCh,size=resol,scaleWeights=self.scaleWeights)) #Generate new noise every forward pass
            else:
                n = noiseInputs[stage]
                self.noise.append(addNoise(outCh,noise=noise,scaleWeights=self.scaleWeights)) #Use the cached noise
        
            #Add the biases
            self.biases.append(torch.zeros(1,outCh).requires_grad_(True)) 
        
        for layerId in range(self.nLayers): #Create the layers from to self.nLayers-1
            layer(kernel=3, layerId=layerId)  

        if self.mode == 'resnet': #Add the only toRGB module in the resnet architecture
            self.toRGB.append(Conv2D(inCh=nf(self.nLayers),outCh=self.outCh, kernelSize=1, scaleWeights=self.scaleWeights))

    def forward(self, y, *args, x = None, **kwargs):
        """
        Forward function.
        y (tensor): the disentangled latent vector
        x (tentsor): the constant input map
        *args, **kwargs: extra arguments for the forward step in the pogressive growing configuration
        """
        if x == None: x = self.cInput
        if self.mode == 'revised':
            return self.forwardProgressive_(x,y,*args,**kwargs)
        elif self.mode == 'skip':
            return self.forwardSkip_(x,y)
        elif self.mode == 'resnet':
            return self.forwardResnet_(x,y)

    def paTerm(self, y, *args, x = None, againstInput = 1, **kwargs):
        """
        pull away term function to call
        y (tensor): the disentangled latent vector
        x (tentsor): the constant input map
        *args, **kwargs: extra arguments for the forward step in the pogressive growing configuration
        """
        if x == None: x = self.cInput
        if self.mode == 'revised':
            return self.paTermProgressive_(x,y,againstInput=againstInput,*args,**kwargs)
        elif self.mode == 'skip' or self.mode == 'resnet':
            return self.paTermStatic_(x,y,againstInput=againstInput)
        
    def applyOneLayer(x, y, layer):
        """
        Apply one layer of the generator to
        the constant tensor c and the latent 
        vector y
        """
        x = self.convs[layer](x,y)
        x = self.noise[layer](x)
        bias = self.biases[layer]
        x = x + bias.view(*bias.shape,1,1)
        x = self.activation(x)
        return x
           
    def forwardProgressive_(self, x, y, layerId = None, fadeWt = 1.):
        """
        Perform a forward pass using
        the progressive growing architecture
        """
        assert layerId % 2 == 0, f'Synthesis Module ERROR: the layer ID in the forward call must be an even integer ({layerID})'
        asser layerId < len(self.convs), f'Synthesis Module ERROR: the layer ID {layerId} is out of bounds {len(self.convs)}'
        if layerId == None:
            layerId = len(self.convs)-1

        for layer in range(layerId-1): #Apply all layers up to layer LayerId-2
                if layer % 2:
                    x = F.interpolate(x, scale_factor=2, mode=self.upsample)
                self.applyOneLayer(x, y, layer)
        
        if fadeWt < 1:   #We are in a fade stage
            prev_x = x #Get the output for the previous resolution
            prev_x = self.toRGB[layerId//2-1](prev_x) #Transform it to RGB
            prev_x = F.interpolate(prev_x, scale_factor=2, mode=self.upsample) 

            x = F.interpolate(x, scale_factor=2, mode=self.upsample)
            x = self.applyOneLayer(x, y, layerId-1) 
            x = self.applyOneLayer(x, y, layerId)   #Compute the output for the current resolution
            x = self.toRGB[layerId//2](x) #Transform it to RGB       
        
            x = fadeWt*x + (1-fadeWt)*prev_x #Interpolate
        
        else:
            x = F.interpolate(x, scale_factor=2, mode=self.upsample)
            x = self.applyOneLayer(x, y, layerId-1) 
            x = self.applyOneLayer(x, y, layerId)   #Compute the output for the current resolution
            x = self.toRGB[layerId//2](x) #Transform it to RGB       

        return x
    
    def forwardSkip_(self, x, y):
        """
        Perform a forward pass using
        the architecture with skip connections
        """"
        output = 0.
        for layer in range(self.nLayers): #Apply all layers
            if layer % 2: #Odd layer, so increase size
                x = F.interpolate(x, scale_factor=2, mode=self.upsample)
                output = F.interpolate(output, scale_factor=2, mode=self.upsample)

            x = self.applyOneLayer(x, y, layer)

            if not layer % 2:  #Even layer, so get the generated output for the given resolution, resize it, and add it to the final output
                output = output + self.toRGB[layer//2](x)
        
        return output

    def forwardResnet_(self, x, y):
        """
        Perform a forward pass using
        the architecture with residual networks
        """"
        carryover = 0.
        for layer in range(self.nLayers): #Apply all layers
            if layer % 2: #Odd layer, so increase size
                x = F.interpolate(x, scale_factor=2, mode=self.upsample)
                carryover = self.lp[layer//2](carryover)
                carryover = F.interpolate(carryover, scale_factor=2, model=self.upsample)

            x = self.applyOneLayer(x, y, layer)

            if not layer % 2:  #Even layer, so add and actualize carryover value
                if carryover: #If there is a carryover, add it to the output
                    x = (carryover + x)/np.sqrt(2)
                carryover = x

        x = self.toRGB[0](x) #Use the only toRGB for this net

        return x
