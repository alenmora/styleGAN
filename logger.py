import torch
import numpy as np
import utils
import os
from datetime import datetime

class Logger:
    """
    Logger class to output net status, images and network snapshots
    """
    def __init__(self, trainer, config):
        self.logPath = utils.createDir(config.logPath)
        self.logStep = int(config.logStep)
        self.saveImageEvery = int(config.saveImageEvery)
        self.saveModelEvery = int(config.saveModelEvery)
        self.trainer = trainer
        self.device = trainer.device
        self.latentSize = int(config.latentSize)
        self.resumeTraining = (config.resumeTraining != None)
        
        self.dontLog = config.deactivateLog
        
        self.samplesStable = trainer.imgStable
        self.samplesFading = trainer.imgFading

        self.nCritPerGen = int(config.nCritPerGen)

        self.cLR = config.cLR
        self.gLR = config.gLR

        self.lamb = config.lamb                     #lambda 
        self.obj = config.obj                       #objective value (1-GP)
        self.epsilon = config.epsilon               #epsilon (drift loss)
        self.paterm = config.paterm                 #Pulling away term

        self.lossFunc = config.lossFunc

        self.loops = config.loops
        self.tick = config.tick
        
        #monitoring parameters
        self.genLoss = 0
        self.criticLoss = 0
        
        self.logCounter = -1
        self.ncAppended = 0
        self.ngAppended = 0

        self.snapCounter = 0
        self.imgCounter = 0

        self.catchLatents = config.returnLatents

        #Outputs 
        self.netStatusHeaderShown = False
        self.archFile = 'architecture.txt'
        self.logFile = 'netStatus.txt'
        self.latentsFile = 'latents.txt'

    def logArchitecture(self):
        """
        This function will print hyperparameters and architecture and save the in the log directory under the architecture.txt file
        """
        if self.dontLog:
            return

        cstFcn = f'Cost function model: {self.lossFunc}\n'
        
        hyperParams = (f'HYPERPARAMETERS - cLR = {self.cLR}|gLR = {self.gLR}|lambda = {self.lamb}|'
                      f'obj = {self.obj}|epsilon = {self.epsilon}|{self.loops} loops, showing {self.tick} images per loop'
                      f'Using pulling away regularization? {f'yes, with value {self.paterm}' if self.paterm != None else 'no'}')
        
        architecture = '\n' + str(self.trainer.crit) + '\n\n' + str(self.trainer.gen) + '\n\n'
        
        print(cstFcn+hyperParams)

        f = os.path.join(self.logPath, self.archFile)

        utils.writeFile(f, cstFcn+hyperParams+architecture, 'w')

    def appendGLoss(self, gloss):
        """
        This function will append the generator loss to the genLoss list
        """
        self.startLogging() #Log according to size of appendGLoss, so call the function when appending
        if self.dontLog:
            return
        self.genLoss = (self.genLoss + gloss).detach().requires_grad_(False)
        self.ngAppended =+ 1

    def appendCLoss(self, closs):
        """
        This function will append the critic training output to the critic lists
        """
        if self.dontLog:
            return
        self.criticLoss = (self.criticLoss + closs).detach().requires_grad_(False)
        self.ncAppended =+ 1

    def startLogging(self):
        
        snapCounter = int(self.trainer.imShown) // self.saveModelEvery
        imgCounter = int(self.trainer.imShown) // self.saveImageEvery

        if snapCounter > self.snapCounter:
            self.saveSnapshot()
            self.snapCounter = snapCounter
        
        if imgCounter > self.imgCounter:
            self.outputPictures()
            self.imgCounter = imgCounter

        if self.dontLog:
            return

        logCounter = int(self.trainer.imShown) // self.logStep
        
        if logCounter > self.logCounter:
            self.logNetStatus()

            #Release memory
            self.genLoss = 0
            self.criticLoss = 0
            self.ncAppended = 0
            self.ngAppended = 0

            torch.cuda.empty_cache()

            self.logCounter = logCounter
        
    def logNetStatus(self):
        """
        Print and write mean losses and current status of net (resolution, stage, images shown)
        """
        if self.netStatusHeaderShown == False:
            colNames = f'time and date |res  |bs   |stage  |fadeWt |iter           |genLoss   |critLoss  '
            sep = '|'.join(['-'*14,'-'*5,'-'*5,'-'*7,'-'*7,'-'*15,'-'*10,'-'*10])
            print(colNames)
            print(sep)

            f = os.path.join(self.logPath,self.logFile)  #Create a new log file
            
            if not self.resumeTraining:
                utils.writeFile(f, colNames, 'w')

            utils.writeFile(f, sep, 'a')

            self.netStatusHeaderShown = True       
      
        res = int(self.trainer.resolurion)
        stage = self.trainer.stage
        fadeWt = self.trainer.fadeWt
        imShown = int(self.trainer.imShown)
        batchSize =self.trainer.dataLoader.batchSize

        # Average all stats and log
        gl = self.genLoss.item()/self.ngAppended if self.ngAppended != 0 else 0.
        cl = self.criticLoss.item()/self.ncAppended if self.ncAppended != 0 else 0.
        
        stats = f' {datetime.now():%H:%M (%d/%m)}'
        stats = stats + "| {:4d}| {:4d}| {:>6s}| {:6.4f}".format(res,batchSize,stage,fadeWt)
        leadingSpaces = 15-len(str(imShown))
        stats = stats + "|"+leadingSpaces*" "+str(imShown)
        stats = stats + "| {:9.4f}| {:9.4f}".format(gl,cl)
        
        print(stats); 
        f = os.path.join(self.logPath,self.logFile)
        utils.writeFile(f, stats, 'a')

    def saveSnapshot(self, title=None):
        """
        Saves model snapshot
        """
        if title == None:
            if self.trainer.stage == 'stable':
                title = f'modelCheckpoint_{self.trainer.res}x{self.trainer.res}_{self.trainer.imShownInRes}_{self.trainer.latentSize}.pth.tar'

                path = os.path.join(self.logPath,title)
                torch.save({'crit':self.trainer.crit.state_dict(), 'cOptimizer':self.trainer.cOptimizer.state_dict(),
                    'gen':self.trainer.gen.state_dict(), 'gOptimizer':self.trainer.gOptimizer.state_dict()}, 
                   path)    

        else:
            title = title+'.pth.tar'
            path = os.path.join(self.logPath,title)
            torch.save({'crit':self.trainer.crit.state_dict(), 'cOptimizer':self.trainer.cOptimizer.state_dict(),
                    'gen':self.trainer.gen.state_dict(), 'gOptimizer':self.trainer.gOptimizer.state_dict()}, 
                   path) 

    def outputPictures(self, size=8):
        """
        outputs real and fake picture samples. If the 
        option of store latents is set to true, it also
        saves the entangled (input latent vector) and
        disentangled (output of the mapping network of the
        generator) to the latents.txt file
        """
        real = self.trainer.getReals(size)
        fake, *latents = self.trainer.getFakes(size)
        stacked = torch.cat([real, fake], dim = 0).cpu()
        fName = '_'.join([str(self.trainer.resolution),str(self.trainer.stage),str(self.trainer.imShownInRes)+'.jpg'])
        path = os.path.join(self.logPath,fName)
        utils.saveImage(stacked, path, nrow = real.size(0))

        if self.catchLatents:
            entangled = latents[1]
            disentangled = latents[0]

            text = fName+'\t'+str(entangled.item())+'\t'+str(disentangled.item())+'\n'
            f = os.path.join(self.logPath, self.latentsFile)

            utils.writeFile(f, text, 'a')