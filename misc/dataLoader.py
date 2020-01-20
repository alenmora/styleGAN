import torch as torch
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader as torchDataLoader
from torchvision.datasets import ImageFolder
import os
from glob import glob
import logging
from PIL import Image

import math

class DataLoader:
    def __init__(self, dataPath = './data/', resolution = None, nCh = None, batchSize = 24, numWorkers = 0):
        self.dataPath = dataPath

        self.ims = glob(os.path.join(self.dataPath,'/*/*.jpg'))
        self.ims += glob(os.path.join(self.dataPath,'/*/*.png'))
        self.ims += glob(os.path.join(self.dataPath,'/*/*.jpeg'))

        assert len(self.ims) > 0, logging.error("dataLoader ERROR: No images found in the given folder")

        if resolution and nCh:
            assert resolution >= 4, logging.error("dataLoader ERROR: The output resolution must be bigger than or equal to 4x4")
            self.resolution = int(resolution)

            assert nCh >= 1, logging.error("dataLoader ERROR: The number of channels must be a positive integer")
            self.nCh = nCh
        else: #deduce resolution from first image in data folder
            firstImg = Image.open(self.ims[0])
            self.resolution = min(firstImg.size)
            self.nCh = len(firstImg.getbands())

        if self.resolution != 2**(int(np.log2(resolution))):
            trueres = 4
            while self.resolution//(trueres*2) != 0:
                trueres = trueres*2

            self.resolution = trueres
            
        self.numWorkers = numWorkers

        self.batchSize = batchSize
        
        self.loadData()

    def loadData(self):
        logging.info(f'Loading data from {self.dataPath} with resolution {self.resolution}x{self.resolution}')
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
            num_workers=self.numWorkers,
            drop_last=True,
            pin_memory = torch.cuda.is_available()
        )

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
        if n is None: n = self.batchSize

        x = self.get_batch()
        for i in range(n // self.batchSize):
            torch.nn.cat([x, self.get_batch()], 0)
        
        return x[:n]
