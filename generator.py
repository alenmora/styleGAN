import torch
from models.generatorNetwork import Generator
from torch import FloatTensor as FT
from misc import utils
import os
import argparse
import math
from config import cfg as opt
import numpy as np

def loadPretrainedWts(dir):
    """
    load trained weights
    """
        
    if os.path.isfile(dir):
        try:
            wtsDict = torch.load(dir, map_location=lambda storage, loc: storage)
            return wtsDict
        except:
            print(f'ERROR: The weights in {dir} could not be loaded')
    else:
        print(f'ERROR: The file {dir} does not exist')    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser('StyleGAN_GEN')
    parser.add_argument('--nImages', type=int, default=20)
    # When sampling the latent vector during training, extreme values are less likely to appear, 
    # and hence the generator is not sufficiently trained in these regions. Hence, we limit the 
    # values of the latent vector to be inside (-psiCut, psiCut)
    parser.add_argument('--psiCut', type=float, default=0.2)        
    parser.add_argument('--latentSize', nargs='?', type=int)
    parser.add_argument('--nChannels', type=int, default=3)
    parser.add_argument('--wtsFile', type=str, default='./pretrainedModels/64x64_modelCheckpoint_semifinal_paterm_nopsicut_nogridtrain_256.pth.tar')
    parser.add_argument('--outputFolder', type=str, default='./generatedImages/')
    parser.add_argument('--outputFile', type=str, nargs='?')
    parser.add_argument('--config', nargs='?', type=str)
    parser.add_argument('--resolution', nargs='?', type=int)
    parser.add_argument('--createInterpolGif', action='store_true')

    args, _ = parser.parse_known_args()

    if args.config:
        opt.merge_from_file(args.config)

    opt.freeze()
    
    endRes = int(args.resolution) if args.resolution else int(args.wtsFile.split('/')[-1].split('x')[0])

    latentSize = args.latentSize if args.latentSize else int(args.wtsFile.split('/')[-1].split('_')[-1].split('.')[0])
    
    device = torch.device('cpu')

    cut = abs(args.psiCut)
    wts = loadPretrainedWts(args.wtsFile)
    n = args.nImages
    folder = utils.createDir(args.outputFolder)
    fname = args.outputFile if args.outputFile else 'generated'
    out = os.path.join(folder, fname+'.png')

    if n <= 0: 
        n = 20

    mopt = opt.model
    gopt = opt.model.gen

    common = {
                    'fmapMax': mopt.fmapMax,
                    'fmapMin': mopt.fmapMin,
                    'fmapDecay': mopt.fmapDecay,
                    'fmapBase': mopt.fmapBase,
                    'activation': mopt.activation,
                    'upsample': mopt.sampleMode,
                    'downsample': mopt.sampleMode
                }

    gen = Generator(**common, **gopt).to(device)

    gen.load_state_dict(wts['gen']) 

    z = utils.getNoise(bs = n, latentSize = latentSize, device = device)
    
    ext_comp = (z.abs() > abs(cut)).type(FT)



    while ext_comp.sum() > 0:
        z = z*(1-ext_comp)+utils.getNoise(bs = n, latentSize = latentSize, device = device)*z*abs(cut)
        ext_comp = (z.abs() > abs(cut)).type(FT)
    
    if cut < 0: z = -z
    
    fakes = gen(z)[0]

    print('single image size: ', str(fakes.shape[2]) + 'x' + str(fakes.shape[2]))
    print(f'number of images: {n}')
    print(f'saving image to: {out}')

    nrows = 1

    if math.sqrt(n) == int(math.sqrt(n)):
        nrows = int(math.sqrt(n))

    elif n > 5:
        i = int(math.sqrt(n))
        while i > 2:
            if (n % i) == 0:
                nrows = i
                break

            i = i-1

    utils.saveImage(fakes, out, nrow=nrows, padding=5)
