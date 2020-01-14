from glob import glob
import os
import torch
import numpy as np 
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch import FloatTensor as FT
from datetime import datetime
import math
import PIL.Image
import animeface
from shutil import copyfile

def resize(x, size):
    transform = transforms.Compose([
        transforms.toPILImage(),
        transforms.Scale(size),
        transforms.ToTensor(),
    ])
    return transform(x)

def writeFile(path, content, mode):
    """
    This will write content to a give file
    """
    file = open(path, mode)
    file.write(content); file.write('\n')
    file.close()

def createDir(dir):
    """
    Create directory
    """
    try: 
        os.makedirs(dir)
        print(f'Created new folder at {dir}')
    except FileExistsError: 
        print(f'Using previously created folder {dir}')
    return dir

def getNoise(bs, latentSize, device):
    """
    This function will return noise
    """
    return FT(bs, latentSize).normal_().to(device=device)

# Loop through each image and process
def makeImagesGrid(tensor, nrow=8, padding=2, pad_value=0):
    if not (torch.is_tensor(tensor) or
            (isinstance(tensor, list) and all(torch.is_tensor(t) for t in tensor))):
        raise TypeError('tensor or list of tensors expected, got {}'.format(type(tensor)))

    # if list of tensors, convert to a 4D mini-batch Tensor
    if isinstance(tensor, list):
        tensor = torch.stack(tensor, dim=0)

    if tensor.dim() == 2:  # single image H x W
        tensor = tensor.view(1, tensor.size(0), tensor.size(1))
    if tensor.dim() == 3:  # single image
        if tensor.size(0) == 1:  # if single-channel, convert to 3-channel
            tensor = torch.cat((tensor, tensor, tensor), 0)
        return tensor
    if tensor.dim() == 4 and tensor.size(1) == 1:  # single-channel images
        tensor = torch.cat((tensor, tensor, tensor), 1)

    # make the mini-batch of images into a grid
    nmaps = tensor.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(tensor.size(2) + padding), int(tensor.size(3) + padding)
    grid = tensor.new(3, height * ymaps + padding, width * xmaps + padding).fill_(pad_value)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            grid.narrow(1, y * height + padding, height - padding)\
                .narrow(2, x * width + padding, width - padding)\
                .copy_(tensor[k])
            k = k + 1
    
    return grid
    
def saveImage(tensor, filename, nrow=8, padding=2, pad_value=0):
    """Save a given Tensor into an image file.
    Args:
        tensor (Tensor or list): Image to be saved. If given a mini-batch tensor,
            saves the tensor as a grid of images by calling ``make_grid``.
        **kwargs: Other arguments are documented in ``make_grid``.
    """
    from PIL import Image
    tensor = tensor.cpu()
    grid = makeImagesGrid(tensor, nrow=nrow, padding=padding, pad_value=pad_value)
    ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy()
    im = Image.fromarray(ndarr)
    im.save(filename)

def switchTrainable(net,isTrainable):
    """
    This is used to switch models parameters to trainable or not
    """
    for p in net.parameters(): p.requires_grad = isTrainable

def debugMemory():
    import collections, gc, resource, torch
    print('maxrss = {}'.format(
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    tensors = collections.Counter((str(o.device), o.dtype, tuple(o.shape))
                                  for o in gc.get_objects()
                                  if torch.is_tensor(o))
    for line in sorted(tensors.items()):
        print('{}\t{}'.format(*line))

def cleanImagesFolder(curPath, newPath, res = None, searchFaces = False, faceThreshold = 0.5):
    """
    Creates a new folder containing all the images
    from the current folder with anime faces on them, using the
    animeface library
    """
    createDir(newPath)

    images = glob(os.path.join(curPath, '*.jpg'))

    for image in images:
        try:
            im = PIL.Image.open(image)
            if res != None:
                if min(im.size) < res: continue
            if searchFaces:
                faces = animeface.detect(im)
                if not faces: continue #Get rid of garbage
                if (faces[0].likelihood < faceThreshold): continue #Get rid of garbage

            imName = image.split('/')[-1]
            newImage = os.path.join(newPath,imName)
            copyfile(image,newImage)
        
        except OSError:
            continue
    