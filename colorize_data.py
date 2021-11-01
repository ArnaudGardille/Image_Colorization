from typing import Tuple
from torch.utils.data import Dataset
import torchvision.transforms as T
#from torchvision import datasets, transforms
import torch
import numpy as np
import os
from os import listdir, walk
from os.path import isfile, join
from pathlib import Path
from skimage import io
import matplotlib.pyplot as plt
from PIL import Image
from skimage.color import lab2rgb, rgb2lab, rgb2gray
from skimage import io

from torchvision.transforms.functional import resize

class ColorizeData(Dataset):
    def __init__(self):
        # Initialize dataset, you may use a second dataset for validation if required
        
        self.path = Path('./landscape_images')
        self.imageNames = [f for f in listdir(self.path) if isfile(join(self.path, f)) and f[-3:] == "jpg"]

        # Use the input transform to convert images to grayscale
        
        self.input_transform = T.Compose([T.ToTensor(),
                                          T.Resize(size=(256,256)),
                                          T.Grayscale()
                                          ])

        self.target_transform = T.Compose([T.ToTensor(),
                                          T.RandomCrop(size=(256,256)),
                                          T.RandomHorizontalFlip(p=0.5)])

    def __len__(self) -> int:
        return len(self.imageNames)
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:

        img_name = self.imageNames[index]
        path = Path.joinpath(self.path, img_name)
        img_original = io.imread(path)
        img_original = self.target_transform(img_original)
        img_original = img_original.permute(1, 2, 0)
        img_original = np.asarray(img_original)

        img_lab = rgb2lab(img_original)
        img_lab = (img_lab + 128) / 255
        img_ab = img_lab[:, :, 1:3]
        img_ab = torch.from_numpy(img_ab.transpose((2, 0, 1))).float()
        
        img_grey = rgb2gray(img_original)
        img_grey = torch.from_numpy(img_grey).unsqueeze(0).float()

        return img_grey, img_ab, img_original


    

