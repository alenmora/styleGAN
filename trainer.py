import torch 
import torch.nn as nn 
import numpy as np
import argparse
from torch.optim import Adam, lr_scheduler
import torch.nn.functional as F
import torch.autograd as autograd
from datetime import datetime
from models.generatorNetwork import Generator
from models.criticNetwork import Critic
from misc.dataLoader import DataLoader
from misc.logger import Logger
from misc import utils
import os
import math
import copy
from random import random

def applyLossScaling(value):
    return value*2**(value)

def undoLossScaling(value):
    return value*2**(-value)

def NonSaturatingLoss(value, truth):
    truth = -1*truth
    return F.softplus(truth*value).mean()

def WassersteinLoss(value, truth):
    truth = -1*truth
    return (truth*value).mean()

class Trainer:
    """
    Trainer class with hyperparams, log, train function etc.
    """
    def __init__(self, opt):
        lopt = opt.logger
        topt = opt.trainer
        mopt = opt.model
        gopt = opt.model.gen
        copt = opt.model.crit
        goopt = opt.optim.gen
        coopt = opt.optim.crit

        #CUDA configuration
        if opt.device == 'cuda' and torch.cuda.is_available():
            os.environ['CUDA_VISIBLE_DEVICES'] = opt.deviceId
            torch.backends.cudnn.benchmark = True
        else:
            opt.device = 'cpu'

        self.device = torch.device(opt.device)
       
        #logger
        self.logger_ = Logger(self, gopt.latentSize, topt.resumeTraining, opt.tick, opt.loops, lopt.logPath, lopt.logStep, 
                                lopt.saveImageEvery, lopt.saveModelEvery, lopt.logLevel, self.device)
        self.logger = self.logger_.logger

        #Logging configuration parameters
        if opt.device == 'cuda':
            num_gpus = len(opt.deviceId.split(','))
            self.logger.info("Using {} GPUs.".format(num_gpus))
            self.logger.info("Training on {}.\n".format(torch.cuda.get_device_name(0)))
            
        #data loader
        dlopt = opt.dataLoader

        self.dataLoader = DataLoader(dlopt.dataPath, dlopt.resolution, dlopt.noChannels, dlopt.batchSize, dlopt.numWorkers)
        
        self.resolution, self.nCh = self.dataLoader.resolution, self.dataLoader.nCh

        # training opt
        assert opt.tick > 0, self.logger.error(f'The number of ticks should be a positive integer, got {opt.tick} instead')
        self.tick = float(opt.tick)

        assert opt.loops > 0, self.logger.error(f'The number of ticks should be a positive integer, got {opt.loops} instead')
        self.loops = int(opt.loops)

        self.imShown = 0
        self.batchShown = self.imShown // self.dataLoader.batchSize

        assert topt.lossFunc in ['NSL','WD'], self.logger.error(f'The specified loss model is not supported. Please choose between "NSL" or "WD"')
        self.lossFunc = topt.lossFunc
        self.criterion = NonSaturatingLoss if self.lossFunc == 'NSL' else WassersteinLoss

        self.applyLossScaling = bool(topt.applyLossScaling)

        self.paterm = int(topt.paterm) if int(topt.paterm) in [0,1,2] else None
        self.lambg = float(topt.lambg)
        self.gLazyReg = max(topt.gLazyReg,1)
        self.styleMixingProb = float(topt.styleMixingProb)

        self.meanPathLength = 0.

        self.plDecay = topt.meanPathLengthDecay

        self.pathRegWeight = topt.pathLengthRWeight

        assert self.paterm is None or not self.styleMixingProb, self.logger.error('Trainer ERROR: The mixing styles regularization is not compatible with a pulling away term')

        assert topt.nCritPerGen > 0, self.logger.error(f'Trainer ERROR: The number of critic training loops per generator loop should be an integer >= 1 (got {topt.nCritPerGen})')
        self.nCritPerGen = int(topt.nCritPerGen)
        
        self.lambR2 = float(topt.lambR2) if topt.lambR2 else 0     #lambda R2
        self.obj = float(topt.obj) if topt.obj else 1              #objective value (1-GP)

        self.lambR1 = float(topt.lambR1) if topt.lambR2 else 0     #lambda R1
        
        self.epsilon = float(topt.epsilon) if topt.epsilon else 0  #epsilon (drift loss)

        self.cLazyReg = max(topt.cLazyReg,1)

        self.kUnroll = int(topt.unrollCritic) if topt.unrollCritic else 0
        
        assert self.kUnroll >= 0, self.logger.error(f'Trainer ERROR: The unroll parameter is less than zero ({self.kUnroll})')

        #Common model parameters
        common = {
                    'fmapMax': mopt.fmapMax,
                    'fmapMin': mopt.fmapMin,
                    'fmapDecay': mopt.fmapDecay,
                    'fmapBase': mopt.fmapBase,
                    'activation': mopt.activation,
                    'upsample': mopt.sampleMode,
                    'downsample': mopt.sampleMode
                }

        #Generator model parameters
        self.gen = Generator(**common, **gopt).to(self.device)
        self.latentSize = self.gen.mapping.latentSize

        self.logger.info(f'Generator constructed. Number of parameters {sum([np.prod([*p.size()]) for p in self.gen.parameters()])}')

        #Critic model parameters
        self.crit = Critic(**mopt, **copt).to(self.device)

        self.logger.info(f'Critic constructed. Number of parameters {sum([np.prod([*p.size()]) for p in self.crit.parameters()])}')

        #Generator optimizer parameters        
        glr, beta1, beta2, epsilon, lrDecay, lrDecayEvery, lrWDecay = list(goopt.values())

        assert lrDecay >= 0 and lrDecay <= 1, self.logger.error('Trainer ERROR: The decay constant for the learning rate of the generator must be a constant between [0, 1]')
        assert lrWDecay >= 0 and lrWDecay <= 1, self.logger.error('Trainer ERROR: The weight decay constant for the generator must be a constant between [0, 1]')
        self.gOptimizer = Adam(filter(lambda p: p.requires_grad, self.gen.parameters()), lr = glr, betas=(beta1, beta2), weight_decay=lrWDecay, eps=epsilon)

        if lrDecayEvery and lrDecay:
            self.glrScheduler = lr_scheduler.StepLR(self.gOptimizer, step_size=lrDecayEvery*self.tick, gamma=lrDecay)
        else:
            self.glrScheduler = None

        self.logger.info(f'Generator optimizer constructed')

        #Critic optimizer parameters
        clr, beta1, beta2, epsilon, lrDecay, lrDecayEvery, lrWDecay = list(coopt.values())

        assert lrDecay >= 0 and lrDecay <= 1, self.logger.error('Trainer ERROR: The decay constant for the learning rate of the critic must be a constant between [0, 1]')
        assert lrWDecay >= 0 and lrWDecay <= 1, self.logger.error('Trainer ERROR: The weight decay constant for the critic must be a constant between [0, 1]')
        
        self.cOptimizer = Adam(filter(lambda p: p.requires_grad, self.crit.parameters()), lr = clr, betas=(beta1, beta2), weight_decay=lrWDecay, eps=epsilon)

        if lrDecayEvery and lrDecay:
            self.clrScheduler = lr_scheduler.StepLR(self.gOptimizer, step_size=lrDecayEvery*self.tick, gamma=lrDecay)
        else:
            self.clrScheduler = None

        self.logger.info(f'Critic optimizer constructed')
        
        self.preWtsFile = opt.preWtsFile
        self.resumeTraining = bool(topt.resumeTraining)
        self.loadPretrainedWts(resumeTraining = self.resumeTraining)

        self.logger.info(f'The trainer has been instantiated.... Starting step: {self.imShown}. Resolution: {self.resolution}')

        self.logArchitecture(clr,glr)

    def logArchitecture(self, clr, glr):
        """
        This function will print hyperparameters and architecture and save the in the log directory under the architecture.txt file
        """
        
        cstFcn = f'Cost function model: {self.lossFunc}\n'
        
        hyperParams = (f'HYPERPARAMETERS - res = {self.resolution}|bs = {self.dataLoader.batchSize}|cLR = {clr}|gLR = {glr}|lambdaR2 = {self.lambR2}|'
                      f'obj = {self.obj}|lambdaR1 = {self.lambR1}|epsilon = {self.epsilon}|{self.loops} loops, showing {self.tick} images per loop'
                      f'|Using pulling away regularization? {f"Yes, with value {self.paterm}" if self.paterm is not None else "No"}')
        
        architecture = '\n' + str(self.crit) + '\n\n' + str(self.gen) + '\n\n'
        
        self.logger.info(cstFcn+hyperParams)

        f = os.path.join(self.logger_.logPath, self.logger_.archFile)

        self.logger.debug(architecture)

        utils.writeFile(f, cstFcn+hyperParams+architecture, 'w')
                        
    def loadPretrainedWts(self, resumeTraining = False):
        """
        Search for weight file in the experiment directory, and loads it if found
        """
        dir = self.preWtsFile
        if os.path.isfile(dir):
            try:
                stateDict = torch.load(dir, map_location=lambda storage, loc: storage)
                self.crit.load_state_dict(stateDict['crit']) 
                self.gen.load_state_dict(stateDict['gen'], strict=False) #Since the cached noise buffers are initialized at None
                self.logger.debug(f'Loaded pre-trained weights from {dir}')
                
                if resumeTraining:
                    self.imShown = stateDict['imShown']
                    self.loops = stateDict['loops']
                    self.tick = stateDict['tick']
                    self.Logger_.genLoss = stateDict['genLoss']
                    self.Logger_.criticLoss = stateDict['criticLoss']
                    self.Logger_.criticLossReals = stateDict['criticLossReals']
                    self.Logger_.criticLossFakes = stateDict['criticLossFakes']
                    self.Logger_.logCounter = stateDict['logCounter']
                    self.Logger_.ncAppended = stateDict['ncAppended']
                    self.Logger_.ngAppended = stateDict['ngAppended']
                    self.Logger_.snapCounter = stateDict['snapCounter']
                    self.Logger_.imgCounter = stateDict['imgCounter']
                    self.cOptimizer.load_state_dict(stateDict['cOptimizer'])
                    self.gOptimizer.load_state_dict(stateDict['gOptimizer'])
                    self.clrScheduler.load_state_dict(stateDict['clrScheduler'])
                    self.glrScheduler.load_state_dict(stateDict['glrScheduler'])
                    self.batchShown = stateDict['batchShown']
                    self.meanPathLength = stateDict['meanPathLength']
                    self.logger.debug(f'And the optimizers states as well')
                
                return True
            except:
                self.logger.error(f'ERROR: The weights in {dir} could not be loaded. Proceding from zero...')
                return False
        else:
            self.logger.error(f'ERROR: The file {dir} does not exist. Proceding from zero...')    
        
        return False

    def getReals(self, n = None):
        """
        Returns n real images
        """ 
        return self.dataLoader.get(n).to(device = self.device)

    def getFakes(self, n = None, z = None):
        """
        Returns n fake images and their latent vectors
        """ 
        if n is None: n = self.dataLoader.batchSize
        
        if z is None:
            z = utils.getNoise(bs = n, latentSize = self.latentSize, device = self.device)
        
            if self.styleMixingProb and random() < self.styleMixingProb:
                zmix = utils.getNoise(bs = n, latentSize = self.latentSize, device = self.device)
                zmix = (zmix - zmix.mean(dim=1, keepdim=True))/(zmix.std(dim=1, keepdim=True)+1e-8)
                output = self.gen(z, zmix = zmix)
        
            else:
                output = self.gen(z)
       
        else:
            output = self.gen(z)
        
        if isinstance(output, list):
            return [*output, z]
        else:
            return [output, z]

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
    
    def R2GradientPenalization(self, reals, fakes):
        alpha = torch.rand(reals.size(0), 1, 1, 1, device=reals.device)
        interpols = (alpha*reals + (1-alpha)*fakes).detach().requires_grad_(True)
        cOut = self.crit(interpols).sum()
        
        if self.applyLossScaling:
            cOut = applyLossScaling(cOut)

        ddx = autograd.grad(outputs=cOut, inputs=interpols,
                              grad_outputs = torch.ones_like(cOut,device=self.device),
                              create_graph = True, retain_graph=True, only_inputs=True)[0]

        ddx = ddx.view(ddx.size(0), -1)

        if self.applyLossScaling:
            ddx = undoLossScaling(ddx)
            
        return ((ddx.norm(dim=1)-self.obj).pow(2)).mean()/(self.obj+1e-8)**2

    def R1GradientPenalization(self, reals):
        reals.requires_grad_(True)
        cOut = self.crit(reals).sum()

        if self.applyLossScaling:
            cOut = applyLossScaling(cOut)

        ddx = autograd.grad(outputs=cOut, inputs=reals,
                              grad_outputs = torch.ones_like(cOut,device=self.device),
                              create_graph = True, retain_graph=True, only_inputs=True)[0]

        ddx = ddx.view(ddx.size(0), -1)

        if self.applyLossScaling:
            ddx = undoLossScaling(ddx)

        return 0.5*(ddx.pow(2).sum(dim=1)).mean()

    def GradientPathRegularization(self, fakes, latents):
        noise = torch.randn_like(fakes) / math.sqrt(fakes.size(2)*fakes.size(3))

        ddx = autograd.grad(outputs=(fakes*noise).sum(), inputs=latents, create_graph=True)[0]

        pathLengths = ddx.norm(dim=1)

        if self.meanPathLength == 0:
            self.meanPathLength = pathLengths.mean()

        else:
            self.meanPathLength = self.meanPathLength + self.plDecay*(pathLengths.mean() - self.meanPathLength)

        self.meanPathLength = self.meanPathLength.detach()

        return (pathLengths - self.meanPathLength).pow(2).mean()
    
    def trainCritic(self):
        """
        Train the critic for one step and store outputs in logger
        """
        utils.switchTrainable(self.crit, True)
        utils.switchTrainable(self.gen, False)

        # real
        real = self.dataLoader.get_batch().to(self.device)
        cRealOut = self.crit(x=real)
        
        # fake
        fake, *_ = self.getBatchFakes()
        cFakeOut = self.crit(x=fake.detach())
        
        lossReals = self.criterion(cRealOut, truth = 1)
        lossFakes = self.criterion(cFakeOut, truth = -1)

        loss = lossReals+lossFakes

        self.crit.zero_grad()
        loss.backward(retain_graph=True); self.cOptimizer.step
        
        if self.batchShown % self.cLazyReg == self.cLazyReg-1:
            extraLoss = 0
            if self.lambR2: 
                extraLoss += self.cLazyReg*self.lambR2*self.R2GradientPenalization(real, fake)
            if self.epsilon: 
                extraLoss += self.epsilon*(cRealOut**2).mean()
            if self.lambR1: 
                extraLoss += self.lambR1*self.R1GradientPenalization(real)

            if extraLoss > 0:
                self.crit.zero_grad()
                extraLoss.backward(); self.cOptimizer.step()
        
        if self.clrScheduler is not None: self.clrScheduler.step() #Reduce learning rate

        self.logger_.appendCLoss(loss, lossReals, lossFakes)

    def trainGenerator(self):
        """
        Train Generator for 1 step and store outputs in logger
        """
        utils.switchTrainable(self.gen, True)
        utils.switchTrainable(self.crit, False)
        
        fake, *latents = self.getBatchFakes()
        cFakeOut = self.crit(fake)

        loss = self.criterion(cFakeOut, truth = 1)

        self.gen.zero_grad()
        loss.backward(retain_graph=True); self.gOptimizer.step() 

        if self.batchShown % self.gLazyReg == self.gLazyReg-1:      
            extraLoss = 0
            if self.pathRegWeight > 0:
                dlatent = latents[0]
                extraLoss = self.GradientPathRegularization(fake, dlatent)
                extraLoss = extraLoss*self.gLazyReg*self.pathRegWeight

            if self.lambg > 0 and self.paterm is not None:
                extraLoss += self.lambg*self.gen.paTerm(dlatent, againstInput = self.paterm)
        
            if extraLoss > 0:
                self.gen.zero_grad()
                extraLoss.backward(); self.gOptimizer.step() 
        
        if self.glrScheduler is not None: self.glrScheduler.step() #Reduce learning rate
                
        self.logger_.appendGLoss(loss)

        return fake.size(0)

    def train(self):
        """
        Main train loop
        """ 

        self.logger.info('Starting training...')   
        self.logger_.startLogging() #Start the  logger

        # loop over images
        while self.imShown < self.tick*self.loops:     
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
            self.batchShown = self.batchShown + 1

            if self.batchShown > max(self.gLazyReg, self.cLazyReg):
                self.batchShown = 0

        self.logger_.saveSnapshot(f'{self.resolution}x{self.resolution}_final_{self.latentSize}')    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="StyleGAN2 pytorch implementation.")
    parser.add_argument('--config', nargs='?', type=str)
    
    args = parser.parse_args()

    from config import cfg as opt
    
    if args.config:
        opt.merge_from_file(args.config)
    
    opt.freeze()
    
    Trainer = Trainer(opt)
    Trainer.train()