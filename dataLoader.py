import torch as torch
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader as torchDataLoader
from torchvision.datasets import ImageFolder
from PIL import Image
import os
from glob import glob
import math

class DataLoader:
    def __init__(self, config):
        self.dataPath = config.dataPath
        if config.resolution:
            assert config.resolution >= 4, 'dataLoader ERROR: The output resolution must be bigger than or equal to 4x4'
            self.endResolution = config.resolution
        else: #deduce resolution from first image in data folder
            ims = glob(os.path.join(self.dataPath,'*/*.jpg'))
            ims += glob(os.path.join(self.dataPath,'*/*.png'))
            ims += glob(os.path.join(self.dataPath,'*/*.jpeg'))
            self.endResolution = min(Image.open(ims[0]).size())

        self.batchSizes = {4:28, 8:28, 16:24, 32:20, 64:16, 128:12, 256:8, 512:6, 1024:4}
        self.resolution = 4
        self.reslevel = 0
        self.resolutions = 2**np.linspace(np.log2(self.resolution, self.endResolution), endpoint=True)
        self.nres = len(self.resolutions)
        self.tickPerStage = math.ceil(config.loops/(2*(nres-1)+nres)) #Same number of ticks for fading and stable stages

        self.nTicks = 0.
        self.tick = config.tick
        self.loops = config.loops
        
        self.num_workers = 0
        
        self.renewData(self.resolution)

    def renewData(self, reslvl):
        print(f'[*] Renew dataloader configuration, load data from {self.dataPath} with resolution {resl}x{resl}')
        self.resolution = int(self.resolutions[reslvl])
        self.batchSize = self.batchSizes[self.resolution]
        self.dataset = ImageFolder(
                                    root=self.dataPath,
                                    transform=transforms.Compose([
                                                    transforms.Resize(size=(self.resolution,self.resolution), interpolation=Image.LANCZOS),
                                                    transforms.RandomHorizontalFlip(),
                                                    transforms.ToTensor(),
                                                    ]))

        self.dataloader = torchDataLoader(
            dataset=self.dataset,
            batch_size=self.batchSize,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory = torch.cuda.is_available()
        )

        return self.resolution

    def __iter__(self):
        return iter(self.dataloader)
    
    def __next__(self):
        return next(self.dataloader)

    def __len__(self):
        return len(self.dataloader.dataset)
   
    def get_batch(self):
        dataIter = iter(self.dataloader)
        return next(dataIter)[0].mul(2).add(-1)     # pixel range [-1, 1]
        
    def get(self, n = None):
        if n == None: n = self.batchSize

        x = self.get_batch()
        for i in range(n // self.batchSize):
            torch.nn.cat([x, self.get_batch()], 0)
        
        return x[:n]
