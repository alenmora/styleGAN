import torch
import torch.nn as nn
import numpy as np 
import torch.autograd as autograd
import math
from generatorBlocks import Mapping, Synthesis
from criticBlocks import Critic
from random import randint

class Generator(nn.Module):
    """
    StyleGAN2 main generator
    Composed of two subnetworks, mapping and synthesis.
    """
    def __init__(self, config):
        super().__init__()
        self.latentSize = config.latentSize
        self.psiCut = config.psiCut
        self.maxCutLayer = config.maxCutLayer
        self.mapping = Mapping(config)
        assert config.synthesisNetwork in ['revised','skip','resnet'], f'Generator ERROR: Invalid synthesis network architecture {config.synthesisNetwork}'
        self.mode = config.synthesisNetwork
        self.synthesis = Synthesis(config)
        self.returnLatents = config.returnLatents
        self.psiCut = config.psiCut
        
    def forward(self, z, *args, zmix = None, wmix = None, **kwargs):
        """
        Forward the generator through the input z
        z (tensor): latent vector
        fadeWt (double): Weight to regularly fade in higher resolution blocks
        zmix (tensor): the second latent vector, used when performing mixing regularization
        wmix (tensor): a second disentangled latent vector, used for style transfer
        """
        assert zmix == None or wmix == None, 'Generator ERROR: You must specify only one between: mixing latent (zmix), or mixing latent disentangled (wmix)'

        w = self.mapping.forward(z)  
        
        w = w.mean(dim=1,keepdim=True)+self.psiCut*(w - w.mean(dim=1,keepdim=True))

        if zmix:
            wmix = self.mapping.forward(zmix)
            wmix = wmix.mean(dim=1,keepdim=True)+self.psiCut*(wmix - wmix.mean(dim=1,keepdim=True))

        if wmix:
            layer = randint(1,self.synthesis.nLayers-1)
            g =self.synthesis.forwardTo(w, layer, *args, **kwargs)
            g = self.synteshis.forwardFrom(wmix, layer, *args, **kwargs)
            return g

        if self.returnLatents:
            return self.synthesis.forward(w, *args, **kwargs), w
        else:
            return self.synthesis.forward(w, *args, **kwargs)

    def paTerm(self, z, againstInput = 1, *args, **kwargs):
        """
        Calculates the pulling away term, as explained in arXiv:1609.03126v4.
        Believed to improve the variance of the generator and avoid mode collapse
        z (tensor): latent vector
        againstInput (int): if 0, the penalty terms will be centered around zero; if 1, around the disentangled latent vectors cosine similariries; if 2, around the square of the cosine similarities
        """
        bs = z.size(0)
        if  bs < 2: #Nothing to do if we only generate one candidate
            return 0
        
        fakes = self.forward(z, *args, **kwargs)
        w = self.mapping.forward(z)
        
        w = w.view(bs, -1) #Unroll
        fakes = fakes.view(bs, -1) #Unroll

        #Calculate pair-wise cosine similarities between batch elements 
        
        suma = 0
        for i in range(bs):
            for j in range(i+1,bs):
                fakesim = torch.nn.functional.cosine_similarity(fakes[i],fakes[j],dim=0)
                if againstInput == 0:
                    suma = suma + fakesim**2
                elif againstInput == 1:
                    zsim = torch.nn.functional.cosine_similarity(w[i],w[j],dim=0)
                    suma = suma + (zsim-fakesim)**2/(ysim**2 +1e-8)
                elif againstInput == 2:
                    ysim = torch.nn.functional.cosine_similarity(w[i],w[j],dim=0)
                    suma = suma + (zsim**2-fakesim**2)**2/(zsim**4 +1e-8)

        return suma/(bs*(bs-1))
