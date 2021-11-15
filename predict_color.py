import argparse
import torch
import os
from colorize_data import ColorizeData
from torch.utils.data import DataLoader
from basic_model import Net
from torch.nn.functional import mse_loss, l1_loss
from torch.optim import SGD, Adam
import matplotlib.pyplot as plt
from time import sleep
from tqdm import tqdm
from torch import save, load
from skimage.color import lab2rgb, rgb2lab, rgb2gray
import numpy as np
from matplotlib.pyplot import cm
from skimage import io
import torchvision.transforms as T
from pathlib import Path

from train import *
parser = argparse.ArgumentParser()
parser.add_argument('-image', type=str, dest="image")

parser.add_argument('-model', type=str,dest="model")


parser.add_argument('-histo', type=bool,dest="histo", default=False)

args = parser.parse_args()


pathImage = Path('./' + args.image)
pathModel = Path('./' + args.model)
pathResult = Path('./' + 'Coloured_' + args.image)

model = torch.load(pathModel).double()
model.eval()

transform = T.Compose([T.ToTensor(),T.Resize(size=(256,256))])


img_grey = io.imread(pathImage, as_gray=True, )
img_grey = transform(img_grey)
img_grey = torch.unsqueeze(img_grey, 0).double()
output_ab = model(img_grey).double()
img_grey = img_grey[0]
output_ab = output_ab[0]
print(output_ab.shape)
output = to_rgb(img_grey, ab_input=output_ab.detach())


plt.imsave(arr=output, fname='Coloured_' + args.image)

if args.histo:
    _ = plt.hist(output.ravel(), bins = 256, color = 'orange', )
    _ = plt.hist(output[:, :, 0].ravel(), bins = 256, color = 'red', alpha = 0.5)
    _ = plt.hist(output[:, :, 1].ravel(), bins = 256, color = 'Green', alpha = 0.5)
    _ = plt.hist(output[:, :, 2].ravel(), bins = 256, color = 'Blue', alpha = 0.5)
    _ = plt.xlabel('Intensity Value')
    _ = plt.ylabel('Count')
    _ = plt.legend(['Total', 'Red_Channel', 'Green_Channel', 'Blue_Channel'])
    plt.savefig('Histo_Coloured_' + args.image)