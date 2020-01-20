import torch
import numpy as np
import os
from datetime import datetime
import logging
from misc import utils



class Logger:
    """
    Logger class to output net status, images and network snapshots
    """
    def __init__(self, trainer, latentSize = 256, resumeTraining = False, tick=1000, loops=6500, 
                logPath='./exp1/', logStep = 10, saveImageEvery = 20, saveModelEvery = 20, logLevel = None):
        
        #Create logger
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.DEBUG)

        #Create console handler and set the logging level
        console = logging.StreamHandler()
        self.logLevel = logging.INFO if logLevel is None else int(logLevel)
        console.setLevel(self.logLevel)
        
        #Create formatter
        formatter = logging.Formatter('[%(levelname)s]\t%(message)s\t(%(filename)s)')
        console.setFormatter(formatter)

        #Add handler to logger        
        self.logger.addHandler(console)

        self.logPath = utils.createDir(logPath)
        self.tick = int(tick)
        self.loops = int(loops)
        self.logStep = int(logStep*self.tick)
        self.saveImageEvery = int(saveImageEvery*self.tick)
        self.saveModelEvery = int(saveModelEvery*self.tick)

        self.trainer = trainer
        self.latentSize = int(latentSize)
        self.resumeTraining = resumeTraining       

        self.z = utils.getNoise(bs = 25, latentSize = self.latentSize, device = torch.device('cpu'))
        
        #monitoring parameters
        self.genLoss = 0
        self.criticLoss = 0
        self.criticLossReals = 0
        self.criticLossFakes = 0
        
        self.logCounter = -1
        self.ncAppended = 0
        self.ngAppended = 0

        self.snapCounter = 0
        self.imgCounter = 0

        #Outputs 
        self.netStatusHeaderShown = False
        self.archFile = 'architecture.txt'
        self.logFile = 'netStatus.txt'
        self.latentsFile = 'latents.txt'

    def appendGLoss(self, gloss):
        """
        This function will append the generator loss to the genLoss list
        """
        self.startLogging() #Log according to size of appendGLoss, so call the function when appending
        if self.logLevel > logging.INFO:
            return
        self.genLoss = (self.genLoss + gloss).detach().requires_grad_(False)
        self.ngAppended =+ 1

    def appendCLoss(self, closs, clossReals, clossFakes):
        """
        This function will append the critic training output to the critic lists
        """
        if self.logLevel > logging.INFO:
            return
        self.criticLoss = (self.criticLoss + closs).detach().requires_grad_(False)
        self.criticLossReals = (self.criticLossReals + clossReals).detach().requires_grad_(False)
        self.criticLossFakes = (self.criticLossFakes + clossFakes).detach().requires_grad_(False)
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

        if self.logLevel > logging.INFO:
            return

        logCounter = int(self.trainer.imShown) // self.logStep
        
        if logCounter > self.logCounter:
            self.logNetStatus()

            #Release memory
            self.genLoss = 0
            self.criticLoss = 0
            self.criticLossReals = 0
            self.criticLossFakes = 0
            self.ncAppended = 0
            self.ngAppended = 0

            torch.cuda.empty_cache()

            self.logCounter = logCounter
        
    def logNetStatus(self):
        """
        Print and write mean losses and current status of net (resolution, stage, images shown)
        """
        if self.netStatusHeaderShown == False:
            colNames = f'time and date |iter     |genLoss   |critLoss  |cLossReal |cLossFake '
            sep = '|'.join(['-'*14,'-'*9,'-'*10,'-'*10,'-'*10,'-'*10])
            self.logger.info(colNames)
            self.logger.info(sep)

            f = os.path.join(self.logPath,self.logFile)  #Create a new log file
            
            if not self.resumeTraining:
                utils.writeFile(f, colNames, 'w')

            utils.writeFile(f, sep, 'a')

            self.netStatusHeaderShown = True       
      
        imShown = int(self.trainer.imShown)
        
        # Average all stats and log
        gl = self.genLoss.item()/self.ngAppended if self.ngAppended != 0 else 0.
        cl = self.criticLoss.item()/self.ncAppended if self.ncAppended != 0 else 0.
        clr = self.criticLossReals.item()/self.ncAppended if self.ncAppended != 0 else 0.
        clf = self.criticLossFakes.item()/self.ncAppended if self.ncAppended != 0 else 0.
        
        stats = f' {datetime.now():%H:%M (%d/%m)}'
        leadingSpaces = 9-len(str(imShown))
        stats = stats + "|"+leadingSpaces*" "+str(imShown)
        stats = stats + "| {:9.4f}| {:9.4f}| {:9.4f}| {:9.4f}".format(gl,cl,clr,clf)
        
        self.logger.info(stats)
        f = os.path.join(self.logPath,self.logFile)
        utils.writeFile(f, stats, 'a')

    def saveSnapshot(self, title=None):
        """
        Saves model snapshot
        """
        if title is None:
            title = f'modelCheckpoint_{int(self.trainer.imShown)}_{self.trainer.latentSize}.pth.tar'
        else:
            title = title+'.pth.tar'

        path = os.path.join(self.logPath,title)
        torch.save({'crit':self.trainer.crit.state_dict(), 
                        'cOptimizer':self.trainer.cOptimizer.state_dict(),
                        'gen':self.trainer.gen.state_dict(), 
                        'gOptimizer':self.trainer.gOptimizer.state_dict(), 
                        'imShown':self.trainer.imShown,
                        'loops':self.loops,
                        'tick':self.tick,
                        'logCounter':self.logCounter,
                        'ncAppended':self.ncAppended,
                        'ngAppended':self.ngAppended,
                        'snapCounter':self.snapCounter,
                        'imgCounter':self.imgCounter,
                        'genLoss':self.genLoss,
                        'criticLoss':self.criticLoss,
                        'criticLossReals':self.criticLossReals,
                        'criticLossFakes':self.criticLossFakes,
                        'batchShown': self.trainer.batchShown,
                    }, path)

    def outputPictures(self):
        """
        outputs a grid of 25 x 25 pictures generated from the same latents
        """
        
        fake = self.trainer.gen(self.z.to(self.trainer.device)).cpu()
        fName = '_'.join([str(int(self.trainer.resolution)),str(int(self.trainer.imShown))+'.jpg'])
        path = os.path.join(self.logPath,fName)
        utils.saveImage(fake, path, nrow = 5)