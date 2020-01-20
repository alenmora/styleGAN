# StyleGAN2 configuration options

from yacs.config import CfgNode as CN
from torch import cuda
import logging

cfg = CN()

############################
#  Global options
############################

cfg.device = 'cuda' if cuda.is_available() else 'cpu'
cfg.deviceId = '0'

cfg.preWtsFile = ""              # File to get the pretrained weights from

cfg.tick = 1000                  #Unit of images shown (to make input compact)
cfg.loops = 6500                 #Total number of training ticks

############################
#  Data options
############################

cfg.dataLoader = CN()

cfg.dataLoader.dataPath = './data/'     # Folder were the training data is stored
cfg.dataLoader.resolution = 64          #Final image resolution. If not specified, gets it from the first image in the training data
cfg.dataLoader.noChannels = 3           #Number of input and output channels. If not specified, gets it from the first image in the training data
cfg.dataLoader.batchSize = 24
cfg.dataLoader.numWorkers = 0

############################
#  Training Options
############################

cfg.trainer = CN()

cfg.trainer.resumeTraining = False     #Wether to resume a previous training. The user must specify the number of images already shown in the last training session
cfg.trainer.lossFunc = 'NSL'           #Loss model used. Default is Non Saturating Loss (NSL). The other options are Wasserstein's Distance (WD) and Logistic
cfg.trainer.applyLossScaling = False   #Wether to scale any loss function before calculating any gradient penalization term or not

cfg.trainer.paterm = -1                #Include a pulling away term in the generator (arXiv =1609.03126v4). The user should specify if the term is as described in the original paper (by passing 0 to the flag), or centered around the similarity (by passing 1) or the squared similarity (by passing 2) of the latent vectors. -1 to deactivate
cfg.trainer.lambg = 1.                 #Weight of the pulling-away term in the generator
cfg.trainer.gLazyReg = 32              #Number of minibatches shown before computing the regularization term for the generator (lazy regularization) 
cfg.trainer.styleMixingProb = 0.9      #Probabilty to mix styles during training

cfg.trainer.nCritPerGen = 1            #Number of critic training loops per generator training loop

cfg.trainer.lambR2 = 0.                #Weight of the extra R2 gradient penalization (0 = Deactivated)
cfg.trainer.obj = 450                  #Objective value for the gradient norm in R2 regularization (arXiv =1704.00028v3)

cfg.trainer.lambR1 = 10.               #Weight of the extra R1 gradient penalization

cfg.trainer.epsilon = 1e-3             #Weight of the loss term related to the magnitud of the real samples' loss from the critic

cfg.trainer.cLazyReg = 32              #Number of minibatches shown before computing the regularization term for the critic (lazy regularization) 

cfg.trainer.unrollCritic = 0           #For an integer greater than 1, it unrolls the critic n steps (arXiv =1611.02163v4)

############################
#  Common model Options
############################

cfg.model = CN()

cfg.model.fmapMax = 256             #Maximum number of channels in a convolutional block
cfg.model.fmapMin = 1               #Minimum number of channels in a convolutional block
cfg.model.fmapBase = 2048           #Parameter to calculate the number of channels in each block = nChannels = max(min(fmapMax, 4*fmapBase/(resolution**fmapDecay), fmapMin)
cfg.model.fmapDecay = 1.           #Parameter to calculate the number of channels in each block = nChannels = max(min(fmapMax, 4*fmapBase/(resolution**fmapDecay), fmapMin)
cfg.model.activation = 'lrelu'      #Which activation function to use for all networks
cfg.model.sampleMode = 'bilinear'   #Algorithm to use for upsampling and downsampling tensors

############################
#  Generator model Options
############################

cfg.model.gen = CN()

cfg.model.gen.psiCut = 0.8                   #Value at which to apply the psi truncation cut in the generator disentangled latent
cfg.model.gen.maxCutLayer = -1               #Maximum generator layer at which to apply the psi cut (-1 = last layer)
cfg.model.gen.synthesisNetwork = 'skip'      #Network architecture for the generator synthesis. The other option is 'resnet'
cfg.model.gen.latentSize = 256               #Size of the latent vector (z)
cfg.model.gen.dLatentSize = 256              #Size of the disentangled latent vector (w)
cfg.model.gen.normalizeLatents = False       #Wether to normalize the latent vector (z) before feeding it to the mapping network
cfg.model.gen.mappingLayers = 4              #Number of mapping layers
cfg.model.gen.neuronsInMappingLayers = 256   #Number of neurons in each of the mapping layers 
cfg.model.gen.randomizeNoise = False         #Wether to randomize noise inputs every time
cfg.model.gen.scaleWeights = False           #Wether to scale the weights for equalized learning

cfg.optim = CN()
############################
#  Gen optimizer Options
############################

cfg.optim.gen = CN()

cfg.optim.gen.lr = 0.003
cfg.optim.gen.beta1 = 0.
cfg.optim.gen.beta2 = 0.99
cfg.optim.gen.eps = 1e-8
cfg.optim.gen.lrDecay =0.1                #Generator learning rate decay constant 
cfg.optim.gen.lrDecayEvery = 2000         #(Approx) Number of ticks shown before applying the decay to the generator learning rate
cfg.optim.gen.lrWDecay = 0.               #Generator weight decay constant

############################
#  Critic model Options
############################

cfg.model.crit = CN()

cfg.model.crit.scaleWeights = True     #Wether to use weight scaling as in PGGAN in the discriminator
cfg.model.crit.network = 'resnet'      #Network architecture for the critic. The other option is 'skip'
cfg.model.crit.stdDevGroupSize = 8     #Size of the groups to calculate the std dev in the last block of the critic

############################
#  Crit optimizer Options
############################

cfg.optim.crit = CN()

cfg.optim.crit.lr = 0.003
cfg.optim.crit.beta1 = 0.
cfg.optim.crit.beta2 = 0.99
cfg.optim.crit.eps = 1e-8
cfg.optim.crit.lrDecay =0.1                #Critic learning rate decay constant 
cfg.optim.crit.lrDecayEvery = 2000         #(Approx) Number of ticks shown before applying the decay to the critic learning rate
cfg.optim.crit.lrWDecay = 0.               #Critic weight decay constant

############################
#  Logging
############################

cfg.logger = CN()

cfg.logger.logPath = './exp1/'        #Folder were the training outputs are stored
cfg.logger.logLevel = logging.INFO    #Use values from logging
cfg.logger.saveModelEvery = 150.      #(Approx) Number of ticks shown before saving a checkpoint of the model
cfg.logger.saveImageEvery = 20.       #(Approx) Number of ticks shown before generating a set of images and saving them in the log directory
cfg.logger.logStep = 5.               #(Approx) Number of ticks shown before writing a log in the log directory

############################
#  Decoder options
############################

cfg.dec = CN()

cfg.dec.network = 'resnet'       #Network architecture for the decoder
cfg.dec.wtsFile = ''             #Trained weights
cfg.dec.useCriticWeights = True  #Initialize as many parameters of the decoder as possible using the critic trained weights
cfg.dec.resumeTraining = False   #Initialize as many parameters of the decoder as possible using the critic trained weights
cfg.dec.batchSize = 40           #Initialize as many parameters of the decoder as possible using the critic trained weights

############################
#  Decoder optimizer Options
############################

cfg.optim.dec = CN()

cfg.optim.dec.lr = 0.003
cfg.optim.dec.beta1 = 0.
cfg.optim.dec.beta2 = 0.99
cfg.optim.dec.eps = 1e-8
cfg.optim.dec.lrDecay =0.1                #Critic learning rate decay constant 
cfg.optim.dec.lrDecayEvery = 2000         #(Approx) Number of ticks shown before applying the decay to the critic learning rate
cfg.optim.dec.lrWDecay = 0.               #Critic weight decay constant
