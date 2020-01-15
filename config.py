# StyleGAN2 configuration options

import argparse, time
parser = argparse.ArgumentParser('StyleGAN2')

############################
#  Paths and inputs
############################

parser.add_argument('--logPath', type=str, default='./styleGAN2/')       # Folder were the training output logs are stored
parser.add_argument('--dataPath', type=str, default='./data/')           # Folder were the training data is stored
parser.add_argument('--preWtsFile', type=str, default=None)               # File to get the pretrained weights from

############################
#  Training parameters
############################

parser.add_argument('--tick', type=int, default=1000)                      #Unit of images shown (to make input compact)
parser.add_argument('--loops', type=int, default=10000)                    #Total number of training ticks
parser.add_argument('--gLR', type=float, default=0.001)                    #Generator learning rate
parser.add_argument('--gLRDecay', type=float, default=0.5)                 #Generator learning rate decay constant 
parser.add_argument('--glRDecayEvery', type=int, default=800)              #(Approx) Number of ticks shown before applying the decay to the generator learning rate
parser.add_argument('--gLRWdecay', type=float, default=0.)                 #Generator weight decay constant
parser.add_argument('--cLR', type=float, default=0.001)                    #Critic learning rate
parser.add_argument('--cLRDecay', type=float, default=0.5)                 #Critic learning rate decay constant 
parser.add_argument('--glRDecayEvery', type=int, default=800)              #(Approx) Number of ticks shown before applying the decay to the critic learning rate
parser.add_argument('--cLRWdecay', type=float, default=0.)                 #Critic weight decay constant
parser.add_argument('--nCritPerGen', type=int, default=1)                  #Number of critic training loops per generator training loop
parser.add_argument('--computeGRegTermsEvery', type=int, default=16)       #Number of minibatches shown before computing the regularization term for the generator (lazy regularization) 
parser.add_argument('--computeCRegTermsEvery', type=int, default=32)       #Number of minibatches shown before computing the regularization term for the critic (lazy regularization) 
parser.add_argument('--gOptimizerBetas', type=str, default='0.0 0.99')     #Generator adam optimizer beta parameters
parser.add_argument('--cOptimizerBetas', type=str, default='0.0 0.99')     #Critic adam optimizer beta parameters
parser.add_argument('--lossFunc', choices=['DWD','NSL'], default='WD')     #Loss model used. Default is Wasserstein's Distance (WD). The other option is NSL (Non Saturating Loss)
parser.add_argument('--lamb', type=float, default=10)                         #Weight of the extra GP loss term (WD) or the R1 (0-centered GP) regularization term (NSL) in the critic loss function

#Extra hyperparameters for the WD loss function (extra terms: gradient penalty and drift loss)
parser.add_argument('--obj', type=float, default=450)                      #Objective value for the gradient norm in GP regularization (arXiv:1704.00028v3)
parser.add_argument('--epsilon', type=float, default=1e-3)                 #Weight of the loss term related to the magnitud of the loss function for the critic

parser.add_argument('--paterm', nargs='?')                                 #Include a pulling away term in the generator (arXiv:1609.03126v4). The user should specify if the term is as described in the original paper (by passing False to the flag), or centered around the distance of the inputs (by passing True)
parser.add_argument('--lambg', type=float, default=1)                      #Weiht of the pulling-away term in the generator
parser.add_argument('--unrollCritic', nargs='?', type=int)                 #For an integer value n greater than 1, it unrolls the critic n steps (arXiv:1611.02163v4)

############################
#  Logging
############################

parser.add_argument('--deactivateLog', action='store_true')       #If passed, there will be no logging
parser.add_argument('--saveModelEvery', type=int, default=100)    #(Approx) Number of ticks shown before saving a checkpoint of the model
parser.add_argument('--saveImageEvery', type=int, default=50)     #(Approx) Number of ticks shown before generating a set of images and saving them in the log directory
parser.add_argument('--logStep', type=int, default=3)             #(Approx) Number of ticks shown before writing a log in the log directory
parser.add_argument('--returnLatents',action='store_true')        #Return, together with the images, a text file with the entangled and disentangled latent vectors

############################
#  Network parameters
############################

parser.add_argument('--fmapMax', type=int, default=256)                                  #Maximum number of channels in convolutional block
parser.add_argument('--fmapMin', type=int, default=1)                                    #Minimum number of channels in convolutional block

# from the equation nchannels = 4*fmapBase/(resolution**fmapDecay). 
# The number of channels of the constant input tensor are determined from
# this equation and the previous cut values as well
parser.add_argument('--fmapBase', type=int, default=2048)                                #Parameter to calculate the number of channels in each block
parser.add_argument('--fmapDecay', type=float, default=1.)                               #Parameter to calculate the number of channels in each block

parser.add_argument('--useWeightScale', type=bool, default=True)                         #Wether to use weight scaling as in PGGAN
parser.add_argument('--psiCut',type=float,default=1.5)                                   #Value at which to apply the psi truncation cut in the generator
parser.add_argument('--maxCutLayer',type=float,default=None)                             #Maximum generator layer at which to apply the psi cut (None = deactivated)
parser.add_argument('--styleMixingProb',nargs='?',type=float)                            #Probabilty to mix styles during training. If not specified, there is no mixing
parser.add_argument('--synthesisNetwork', choices=['revised','skip','resnet'], default='skip')  #Network architecture for the generator synthesis
parser.add_argument('--criticNetwork', choices=['revised','skip','resnet'], default='resnet')   #Network architecture for the critic 
parser.add_argument('--stdDevGroup', type=int, default=8)                  #Size of the groups to calculate the std dev in the last block of the critic
parser.add_argument('--latentSize', type=int, default=256)                 #Size of the latent vector
parser.add_argument('--dLatentSize', type=int, default=256)                #Size of the disentangled (W) latent vector
parser.add_argument('--mappingLayers',type=int,default=4)                  #Number of mapping layers
parser.add_argument('--neuronsInMappingLayer',type=int,default=256)        #Number of neurons in each of the mapping layers 
parser.add_argument('--normalizeLatents', action='store_true')             #Wether to normalize the latent vector (Z) before feeding it to the mapping network
parser.add_argument('--resolution', type=int, nargs='?')                   #Final image resolution. If not specified, gets it from the first image in the training set
parser.add_argument('--randomizeNose', action='store_true')                                              #Wether to randomize noise inputs every time
parser.add_argument('--activationFunction', choices=['lrelu','relu','tanh','sigmoid'], default='lrelu')  #Which activation function to use for the whole network
parser.add_argument('--outputChannels', type=int, default=3)
parser.add_argument('--upsampleMode', type=str, default='bilinear')         #Algorithm to use for upsampling and downsampling tensors

############################
#  Resume training
############################
parser.add_argument('--resumeTraining', nargs='*')  #Resumes a previous training. The user must specify the current loop number (uses the weights file from --preWtsFile)

##Parse and save configuration
config, _ = parser.parse_known_args()

############################
#  Decoder options
############################
parser.add_argument('--decoderNetwork', choices=['revised','skip','resnet'], default='resnet')  #Network architecture for the decoder