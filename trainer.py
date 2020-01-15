import torch.nn as nn 
import torch 
import numpy as np
from torch.optim import Adam, lr_scheduler
from datetime import datetime
from config import config 
import utils
from generatorNetwork import Generator
from criticNetwork import Critic
from dataLoader import DataLoader
from logger import Logger
import os
import math
import copy
from random import random

def nonSaturatingLossG_(fakeScores):
    return -torch.log(fakeSkores).mean()

def nonSaturatingLossC_(realScores, fakeScores):
    return -torch.log(1-fakeSkores).mean()-log(realScores).mean()

def wassersteinLossG_(fakeScores):
    return -fakeScores.mean()

def wassersteinLossC_(realScores, fakeScores):
    return fakeScores.mean()-realCores.mean()

def gradientPenalization(gradient, obj):
    return ((gradInterpols.norm(dim=1)-obj)**2).mean()/(obj+1e-8)**2

def R1GradientPenalization(gradient): # arXiv 1801.04406
    return 2*((gradInterpols.norm(dim=1))**2).mean()

def driftLoss(realScores):
    return (realScores**2).mean()

def getLossFunctions(name):
    if name == 'NSL':
        return nonSaturatingLossC_, nonSaturatingLossG_
    elif name == 'WGP':
        return wassersteinLossC_, wassersteinLossG_


class Trainer:
    """
    Trainer class with hyperparams, log, train function etc.
    """
    def __init__(self, config):

        #CUDA configuration parameters
        self.useCuda = torch.cuda.is_available()
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            print('Using CUDA...')
        else:
            self.device = torch.device('cpu')
        
        #Config
        self.config = config

        #data loading
        self.dataLoader = dataLoader(config)
        
        # Hyperparams
        self.nCritPerGen = int(config.nCritPerGen)
        assert self.nCritPerGen > 0, f'Trainer ERROR: The number of critic training loops per generator loop should be an integer >= 1 (got {self.nCritPerGen})'
        self.cLR=config.cLR; self.gLR=config.gLR
        self.latentSize = int(config.latentSize)
                
        #Training parameters
        self.nLoops = config.loops
        self.tick = max(config.tick,1)
        self.imgStable = math.ceil(self.tick*self.nLoops/(2*self.dataLoader.nres+1))
        self.imgFading = math.ceil(self.tick*self.nLoops/(2*self.dataLoader.nres+1))
        
        self.reslvl = 0
        self.resolution = 4
        
        self.imShown = 0.
        self.imShownInRes = 0.
        self.batchShown = 0

        self.endResolution = config.resolution
        self.fadeWt = 1.
        self.kUnroll = 0
        
        if config.unrollCritic:
            self.kUnroll = int(config.unrollCritic)
        
        assert self.kUnroll >= 0, f'Trainer ERROR: The unroll parameter is less than zero ({self.kUnroll})'

        # Loss function of critic
        self.lamb = config.lamb              #lambda 
        self.obj = config.obj                #objective value (1-GP)
        self.epsilon = config.epsilon        #epsilon (drift loss)

        self.lossFunc = config.lossFunc
        
        self.critLoss, self.genLoss = getLossFunctions(self.lossFunc) 

        self.lazyRegCritic = max(config.computeCRegTermsEvery,1)

        #Loss function of generator
        self.paterm = config.paterm
        self.lambg = config.lambg
        self.styleMixingProb = config.styleMixingProb

        assert self.paterm == None or self.styleMixingProb == None, 'Trainer ERROR: The mixing styles regularization is not compatible with a pulling away term'
        assert self.paterm == None or self.styleMixingProb == None, 'Trainer ERROR: The mixing styles regularization is not compatible with a pulling away term'
        assert self.styleMixingProb == None or config.returnLatents == False, 'Trainer ERROR: It is not possible to return the latents while performing mixing regularization'

        self.lazyRegGenerator = max(config.computeGRegTermsEvery,1)

        # models
        self.createModels()

        # Optimizers
        assert config.gLRDecay > 0 and config.gLRDecay <= 1, 'Trainer ERROR: The decay constant for the learning rate of the generator must be a constant between [0, 1]'
        self.gLRDecay =config.gLRDecay

        betas = config.gOptimizerBetas.split(' ')
        beta1, beta2 = float(betas[0]), float(betas[1])
        assert config.gLRWdecay >= 0 and config.gLRWdecay <= 1, 'Trainer ERROR: The weight decay constant for the generator must be a constant between [0, 1]'
        self.gOptimizer = Adam(filter(lambda p: p.requires_grad, self.gen.parameters()), lr = self.gLR, betas=(beta1, beta2), weight_decay=config.gLRWdecay)

        self.glrScheduler = lr_scheduler.LambdaLR(self.gOptimizer,lambda epoch: self.gLRDecay)

        assert config.cLRDecay > 0 and config.cLRDecay <= 1, 'Trainer ERROR: The decay constant for the learning rate of the critic must be a constant between [0, 1]'
        self.cLRWecay =config.cLRDecay

        assert config.cLRWdecay >= 0 and config.cLRWdecay <= 1, 'Trainer ERROR: The weight decay constant for the critic must be a constant between [0, 1]'
        betas = config.cOptimizerBetas.split(' ')
        beta1, beta2 = float(betas[0]), float(betas[1])
        self.cOptimizer = Adam(filter(lambda p: p.requires_grad, self.crit.parameters()), lr = self.cLR, betas=(beta1, beta2), weight_decay=config.cLRWdecay)

        self.clrScheduler = lr_scheduler.LambdaLR(self.cOptimizer,lambda epoch: self.cLRDecay)
                
        # Paths        
        self.preWtsFile = config.preWtsFile
        
        if config.resumeTraining:
            self.resumeTraining(config)
        
        elif self.preWtsFile: self.loadPretrainedWts()
        
        #Log
        self.logger = logger(self, config)
        self.logger.logArchitecture()

        print(f'The trainer has been instantiated.... Starting step: {self.nTicks}. Start resolution: {self.res}. Final resolution: {self.endRes}')
        
    def resumeTraining(self, config):
        """
        Resumes the model training, if a valid pretrained weights file is given, and the 
        starting resolution and number of images shown for the current resolution are correctly
        specified
        """
        if not self.loadPretrainedWts():
            print('Could not load weights for the pretrained model. Starting from zero...')
            return

        res, imShownInRes = config.resumeTraining
        res, imShownInRes = int(res), int(imShownInRes)
        curResLevel = 0
            
        curResLevel = int(np.log2(res)-2)
        
        if 2**(curResLevel+2) != res or res > 1024 or res < 4:
            print(f'Trainer ERROR: the current resolution ({res}) is not a power of 2 between 4 and 1024. Proceeding from zero...')
            return
 
        if curResLevel > 0 and imShownInRes < self.imgFading:
            print(f'Trainer ERROR: the training resuming can only proceed from stable phases. However, the number of images already shown ({imShownInRes}) '
                  f'is less than the number of images shown in the fading stage ({self.imgFading}). Proceeding from zero...')

        else:
            if curResLevel > 0:
                if imShownInRes > self.imgFading+self.imgStable:
                    print(f'Trainer ERROR: the number of images already shown ({imShownInRes}) is higher than the number of images shown for one resolution '
                          f'level ({self.imgFading+self.imgStable})! Proceeding from zero...')
                else:
                    self.imShown = curResLevel*(self.imgFading+self.imgStable) + imShownInRes
                    self.imShownInRes = imShownInRes
                    self.resolution = res
                    self.reslvl = int(curResLevel)+1
                    self.dataLoader.renewData(self.reslvl)
                    for i in range(curResLevel):
                        self.clrScheduler.step()
                        self.glrScheduler.step()
            else:
                if imShownInRes > self.imgStable:
                    print(f'Trainer ERROR: the number of images already shown ({imShownInRes}) is higher than the number of images shown for the first ' 
                          f'resolution ({self.imgStable})! Proceeding from zero...')
                else:
                    self.imShown = imShownInRes
                    self.imShownInRes = imShownInRes
                    self.resolution = res
                    self.reslvl = int(curResLevel)+1
                    self.dataLoader.renewData(self.reslvl)
                        
    def createModels(self):
        """
        This function will create models and their optimizers
        """
        self.gen = Generator(self.config).to(self.device)
        self.crit = Critic(self.config).to(self.device)
        
        print('Models Instantiated. # of trainable parameters Critic: %e; Generator: %e' 
              %(sum([np.prod([*p.size()]) for p in self.crit.parameters()]), 
                sum([np.prod([*p.size()]) for p in self.gen.parameters()])))
        
    def loadPretrainedWts(self):
        """
        Search for weight file in the experiment directory, and loads it if found
        """
        dir = self.preWtsFile
        if os.path.isfile(dir):
            try:
                wtsDict = torch.load(dir, map_location=lambda storage, loc: storage)
                self.crit.load_state_dict(wtsDict['crit']) 
                self.gen.load_state_dict(wtsDict['gen'])
                self.cOptimizer.load_state_dict(wtsDict['cOptimizer'])
                self.gOptimizer.load_state_dict(wtsDict['gOptimizer'])
                print(f'Loaded pre-trained weights from {dir}')
                return True
            except:
                print(f'ERROR: The weights in {dir} could not be loaded. Proceding from zero...')
                return False
        else:
            print(f'ERROR: The file {dir} does not exist. Proceding from zero...')    
        
        return False

    def getReals(self, n = None):
        """
        Returns n real images
        """ 
        return self.dataLoader.get(n).to(device = self.device)

    def getFakes(self, n = None):
        """
        Returns n fake images and their latent vectors
        """ 
        if n == None: n = self.dataLoader.batchSize
        z = utils.getNoise(bs = n, latentSize = self.latentSize, device = self.device)

        if random() < self.styleMixingProb:
            zmix = utils.getNoise(bs = n, latentSize = self.latentSize, device = self.device)
            return self.gen(z, zmix = z2, maxLayer = self.reslvl, fadeWt = self.fadeWt)


        return *self.gen(z, maxLayer = self.reslvl, fadeWt=self.fadeWt), z

    def getBatchReals(self):
        """
        Returns a batch of real images
        """ 
        return self.dataLoader.get_batch().to(device = self.device)

    def getBatchFakes(self):
        """
        Returns a batch of fake images and the latent vector which generated it
        """
        return self.getFakes()
    
    def trainCritic(self):
        """
        Train the critic for one step and store outputs in logger
        """
        self.cOptimizer.zero_grad()
        utils.switchTrainable(self.crit, True)
        utils.switchTrainable(self.gen, False)

        maxLayer = 2*self.reslvl+1

        # real
        real = self.dataLoader.get_batch()
        cRealOut = self.crit(x=real, maxLayer = maxLayer, fadeWt=self.fadeWt)
        
        # fake
        fake, *_ = self.getBatchFakes()
        cFakeOut = self.crit(x=fake.detach(), maxLayer = maxLayer, fadeWt=self.fadeWt)
        
        loss = self.critLoss(cRealOut, cFakeOut)
        
        if self.lossFunc == 'WGP' and self.batchShown % self.lazyRegCritic == self.lazyRegCritic-1:
            alpha = torch.rand(real.size(0), 1, 1, 1, device=self.device)
            interpols = (alpha*real + (1-alpha)*fake).detach().requires_grad_(True)
            gradInterpols = self.crit.getGradientsWrtInputs(interpols, maxLayer = maxLayer, fadeWt=self.fadeWt)
            loss += self.lamb*gradientPenalization(gradInterpols,self.obj)
            loss += self.epsilon*driftLoss(cRealOut)

        elif self.lossFunc == 'NSL' and self.batchShown % self.lazyRegCritic == self.lazyRegCritic-1:
            grads = self.crit.getGradientsWrtInputs(real, maxLayer = maxLayer, fadeWt=self.fadeWt) 
            loss += self.lamb*R1GradientPenalization(grads)

        loss.backward(); self.cOptimizer.step()

        self.logger.appendCLoss(loss)
        
    def trainGenerator(self):
        """
        Train Generator for 1 step and store outputs in logger
        """
        self.gOptimizer.zero_grad()
        utils.switchTrainable(self.gen, True)
        utils.switchTrainable(self.crit, False)

        maxLayer = 2*self.reslvl+1
        
        fake, *latents = self.getBatchFakes()
        cFakeOut = self.crit(x=fake, fadeWt=self.fadeWt, maxLayer = maxLayer)
        
        loss = self.genLoss(cFakeOut)

        if self.paterm != None and self.batchShown % self.lazyRegGenerator == self.lazyRegGenerator-1:
            latent = latents[0]
            loss += self.lambg*self.gen.paTerm(latent, againstInput = self.paterm, maxLayer = maxLayer, fadeWt = self.fadeWt)
        
        loss.backward(); self.gOptimizer.step()
        
        self.logger.appendGLoss(loss)

        return fake.size(0)

    def train(self):
        """
        Main train loop
        """ 

        print('Starting training...')   
        self.logger.startLogging() #Start the logging

        # Loop over the first resolution, which only has stable stage. Since each batch shows batchSize images, 
        # we need only samplesWhileStable/batchSize loops to show the required number of images
        if self.reslvl == 0:
            self.stage = 'stable'
            while self.imShownInRes < self.imgStable:
                self.doOneTrainingStep() #Increases self.imShownInRes and self.imShown
                
            self.imShownInRes = 0 #Reset the number of images shown at the end (to allow for training resuming)

        # loop over resolutions 8 x 8 and higher
        while self.imShown < self.tick*self.nLoops:   
            self.fadeWt = min(float(self.imShownInRes)/self.imgFading, 1)
            self.stage = 'fade' if self.fadeWt < 1 else 'stable'

            if self.fadeWt == 0: #We just began to fade. So we need to increase the resolutions
                self.clr_scheduler.step() #Reduce learning rate
                self.glr_scheduler.step() #Reduce learning rate

                self.reslvl = self.reslvl+1
                self.resolution = int(self.dataLoader.renewData(self.reslvl))
                
            self.doOneTrainingStep() #Increases self.imShownInRes and self.imShown

            if self.imShownInRes > self.imgStable+self.imgFading:
                self.imShownInRes = 0 #Reset the number of images shown at the end (to allow for training resuming) 

        self.logger.saveSnapshot(f'{self.res}x{self.res}_final_{self.latentSize}')
            
    def doOneTrainingStep(self):
        """
        Performs one train step for the generator, and nCritPerGen steps for the critic
        """ 
        if self.kUnroll:
            for i in range(self.nCritPerGen):
                self.trainCritic()
                if i == 0:
                    self.cBackup = copy.deepcopy(self.crit)
        else:                
            for i in range(self.nCritPerGen):
                self.trainCritic()
        
        shown = self.trainGenerator() #Use the generator training batches to count for the images shown, not the critic
        
        if self.kUnroll:
            self.crit.load(self.cBackup)

        self.imShown = self.imShown + int(shown) 
        self.imShownInRes = self.imShownInRes + int(shown)
        self.batchShown = self.batchShown + 1

        if self.batchShown > max(self.lazyRegGenerator, self.lazyRegCritic):
            self.batchShown = 0

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True           # boost speed.
    Trainer = Trainer(config)
    Trainer.train()