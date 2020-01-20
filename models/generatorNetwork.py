import torch
import torch.nn as nn
import numpy as np 
import math
from models.generatorBlocks import Mapping, Synthesis
from random import randint

class Generator(nn.Module):
    """
    StyleGAN2 main generator
    Composed of two subnetworks, mapping and synthesis.
    """
    def __init__(self, latentSize = 256, dLatentSize = 256, mappingLayers = 4, neuronsInMappingLayers = 256, normalizeLatents = True,
                resolution = 64, fmapBase = 2048, fmapDecay = 1, fmapMax = 256, fmapMin = 1, randomizeNoise = False, 
                activation = 'lrelu', scaleWeights = False, outCh = 3, upsample = 'bilinear', synthesisMode = 'skip', psiCut = 0.7,
                maxCutLayer = -1, **kwargs):
        
        super().__init__()
        self.mapping = Mapping(latentSize = latentSize, dLatentSize = dLatentSize, mappingLayers = mappingLayers, 
                                neuronsInMappingLayers = neuronsInMappingLayers, activation = activation, 
                                scaleWeights = scaleWeights, normalizeLatents = normalizeLatents)
        
        
        self.synthesis = Synthesis(dLatentSize = dLatentSize, resolution = resolution, fmapBase = fmapBase, fmapDecay = fmapDecay, fmapMax = fmapMax, 
                                fmapMin = fmapMin, randomizeNoise = randomizeNoise, activation = activation, scaleWeights = scaleWeights, outCh = 3, 
                                upsample = upsample, mode = synthesisMode)
        
        self.psiCut = psiCut
        self.maxCutLayer = self.synthesis.nLayers-1 if maxCutLayer < 0 else maxCutLayer
        
    def forward(self, z, *args, zmix = None, wmix = None, cutLayer = None, **kwargs):
        """
        Forward the generator through the input z
        z (tensor): latent vector
        fadeWt (double): Weight to regularly fade in higher resolution blocks
        zmix (tensor): the second latent vector, used when performing mixing regularization
        wmix (tensor): a second disentangled latent vector, used for style transfer
        cutLayer (int): layer at which to introduce the new mixing element
        """
        assert zmix is None or wmix is None, 'Generator ERROR: You must specify only one between: mixing latent (zmix), or mixing latent disentangled (wmix)'

        w = self.mapping.forward(z)  
        
        w = w.mean(dim=1,keepdim=True)+self.psiCut*(w - w.mean(dim=1,keepdim=True))

        if zmix is not None:
            wmix = self.mapping.forward(zmix)
            wmix = wmix.mean(dim=1,keepdim=True)+self.psiCut*(wmix - wmix.mean(dim=1,keepdim=True))

        if wmix is not None:
            if cutLayer is None:
                cutLayer = self.maxCutLayer
            layer = randint(1,cutLayer)
            x, extraOutput =self.synthesis.forwardTo(w, layer, *args, **kwargs)
            g = self.synthesis.forwardFrom(wmix, x, extraOutput, layer, *args, **kwargs)
            return g

        return [self.synthesis.forward(w, *args, **kwargs), w]
        
    def paTerm(self, z, *args, againstInput = 1, **kwargs):
        """
        Calculates the pulling away term, as explained in arXiv:1609.03126v4.
        Believed to improve the variance of the generator and avoid mode collapse
        z (tensor): latent vector
        againstInput (int): if 0, the penalty terms will be centered around zero; if 1, around the disentangled latent vectors cosine similariries; if 2, around the square of the cosine similarities
        """
        bs = z.size(0)
        if  bs < 2: #Nothing to do if we only generate one candidate
            return 0
        
        w = self.mapping.forward(z)
        fakes = self.synthesis.forward(z, *args, **kwargs)
        nCh = fakes.size(1)
        
        w = w.view(bs, -1) #Unroll
        fakes = fakes.view(bs, nCh, -1)  #N x nCh x (h*w)

        #Calculate pair-wise cosine similarities between batch elements        
        suma = 0
        for i in range(bs):
            for j in range(i+1,bs):
                fakesim = torch.nn.functional.cosine_similarity(fakes[i],fakes[j],dim=0).mean()
                if againstInput == 0:
                    suma = suma + fakesim**2
                elif againstInput == 1:
                    wsim = torch.nn.functional.cosine_similarity(w[i],w[j],dim=0)
                    suma = suma + (wsim-fakesim)**2
                elif againstInput == 2:
                    wsim = torch.nn.functional.cosine_similarity(w[i],w[j],dim=0)
                    suma = suma + (wsim**2-fakesim**2)**2

        return suma/(bs*(bs-1))
