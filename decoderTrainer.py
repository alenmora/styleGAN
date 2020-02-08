import torch.nn as nn 
import torch 
import numpy as np
from torch.optim import Adam, lr_scheduler
from misc import utils
from misc.logger import DecoderLogger
from models.generatorNetwork import Generator
from models.decoderNetwork import Decoder
import os
import math
import copy
from random import random
import argparse

class DecoderTrainer:
    """
    Trainer class for the decoder
    """
    def __init__(self, config):
        lopt = opt.logger
        topt = opt.trainer
        mopt = opt.model
        gopt = opt.model.gen
        copt = opt.model.crit
        dopt = opt.dec
        doopt = opt.optim.dec
       
        #logger
        self.logger_ = DecoderLogger(self, gopt.latentSize, dopt.resumeTraining, opt.tick, opt.loops, lopt.logPath, lopt.logStep, 
                                lopt.saveModelEvery, lopt.logLevel)
        self.logger = self.logger_.logger

        #CUDA configuration parameters
        if opt.device == 'cuda':
            os.environ['CUDA_VISIBLE_DEVICES'] = opt.deviceId
            num_gpus = len(opt.deviceId.split(','))
            self.logger.info("Using {} GPUs.".format(num_gpus))
            self.logger.info("Training on {}.\n".format(torch.cuda.get_device_name(0)))
            torch.backends.cudnn.benchmark = True
        self.device = torch.device(opt.device)

        # training opt
        assert opt.tick > 0, self.logger.error(f'The number of ticks should be a positive integer, got {opt.tick} instead')
        self.tick = float(opt.tick)

        assert opt.loops > 0, self.logger.error(f'The number of ticks should be a positive integer, got {opt.loops} instead')
        self.loops = int(opt.loops)

        self.imShown = 0

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

        #Decoder model parameters
        copt.network = dopt.network
        self.decoder = Decoder(**mopt, **copt).to(self.device)

        self.logger.info(f'Decoder constructed. Number of parameters {sum([np.prod([*p.size()]) for p in self.dec.parameters()])}')

        #Decoder optimizer parameters
        clr, beta1, beta2, epsilon, lrDecay, lrDecayEvery, lrWDecay = list(doopt.values())

        assert lrDecay >= 0 and lrDecay <= 1, self.logger.error('Trainer ERROR: The decay constant for the learning rate of the critic must be a constant between [0, 1]')
        assert lrWDecay >= 0 and lrWDecay <= 1, self.logger.error('Trainer ERROR: The weight decay constant for the critic must be a constant between [0, 1]')
        
        self.dOptimizer = Adam(filter(lambda p: p.requires_grad, self.crit.parameters()), lr = clr, betas=(beta1, beta2), weight_decay=lrWDecay, eps=epsilon)

        if lrDecayEvery and lrDecay:
            self.dlrScheduler = lr_scheduler.StepLR(self.gOptimizer, step_size=lrDecayEvery, gamma=lrDecay)
        else:
            self.dlrScheduler = None

        self.logger.info(f'Decoder optimizer constructed')

        #Trained data loading
        dir = dopt.wtsFile
        if os.path.isfile(dir):
            try:
                stateDict = torch.load(dir, map_location=lambda storage, loc: storage)
                self.gen.load_state_dict(stateDict['gen'], strict=False) #Since the cached noise buffers are initialized at None
                self.logger.info(f'Loaded generator trained weights from {dir}')
                
                if dopt.useCriticWeights or dopt.resumeTraining:
                    if 'dec' in stateDict.keys():
                        self.decoder.load_state_dict(stateDict['dec']) #First, try to load a decoder dictionary
                    else:
                        self.decoder.load_state_dict(stateDict['crit'], strict=False) #Last layer won't match, so make strict = False      
                    
                    self.logger.info(f'Loaded critic trained weights from {dir}')

                if dopt.resumeTraining:
                    self.imShown = stateDict['imShown']
                    self.loops = stateDict['loops']
                    self.tick = stateDict['tick']
                    self.Logger_.loss = stateDict['dLoss']
                    self.Logger_.logCounter = stateDict['logCounter']
                    self.Logger_.appended = stateDict['appended']
                    self.Logger_.snapCounter = stateDict['snapCounter']
                    self.Logger_.diffCounter = stateDict['diffCounter']
                    self.dOptimizer.load_state_dict(stateDict['dOptimizer'])
                    self.logger.info(f'And the optimizers states as well')
                
            except:
                self.logger.error(f'ERROR: The information in {dir} could not be loaded. Exiting')
                raise IOError
        else:
            self.logger.error(f'ERROR: The file {dir} does not exist. Proceding Exiting')    
            raise IOError

        utils.switchTrainable(self.gen, False)

        self.batchSize = max(dopt.batchSize, 1)
        
        self.logger.info(f'The trainer has been instantiated...')

    def getBatch(self):
        """
        Returns n fake images and their latent vectors
        """ 
        z = utils.getNoise(bs = self.batchSize, latentSize = self.latentSize, device = self.device)

        return self.gen(z)

    def decoderLoss(self, dout, w):
        return (dout-w).norm(dim=1).mean()

    def trainDecoder(self):
        """
        Train the critic for one step and store outputs in logger
        """
        self.dOptimizer.zero_grad()
        
        # fake
        ims, w = self.getBatch()
        dout = self.decoder(ims)
        
        loss = self.decoderLoss(dout, w)

        loss.backward(); self.dOptimizer.step()

        if self.dlrScheduler is not None: self.dlrScheduler.step() #Reduce learning rate
        
        self.logger.appendLoss(loss)
        
    def train(self):
        """
        Main train loop
        """ 

        print('Starting training...')   
        self.logger.startLogging() #Start the logging

        while self.imShown < self.tick*self.nLoops: self.trainDecoder()

        self.logger.saveSnapshot(f'{self.res}x{self.res}_final_{self.latentSize}_decoder')
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="StyleGAN2 pytorch implementation.")
    parser.add_argument('--config', nargs='?', type=str)
    
    args = parser.parse_args()

    from config import cfg as opt
    
    if args.config:
        opt.merge_from_file(args.config)
    
    opt.freeze()

    Trainer = DecoderTrainer(opt)
    
    Trainer.train()