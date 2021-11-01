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

import warnings
warnings.filterwarnings("ignore")

    
def get_histogram(image):
    
    _ = plt.hist(image.ravel(), bins = 256, color = 'orange', )
    _ = plt.hist(image[:, :, 0].ravel(), bins = 256, color = 'red', alpha = 0.5)
    _ = plt.hist(image[:, :, 1].ravel(), bins = 256, color = 'Green', alpha = 0.5)
    _ = plt.hist(image[:, :, 2].ravel(), bins = 256, color = 'Blue', alpha = 0.5)
    _ = plt.xlabel('Intensity Value')
    _ = plt.ylabel('Count')
    _ = plt.legend(['Total', 'Red_Channel', 'Green_Channel', 'Blue_Channel'])
    plt.show()


def to_rgb(grayscale_input, ab_input):
    '''Show/save rgb image from grayscale and ab channels
     Input save_path in the form {'grayscale': '/path/', 'colorized': '/path/'}'''
    plt.clf() # clear matplotlib 
    color_image = torch.cat((grayscale_input, ab_input), 0).numpy() # combine channels
    color_image = color_image.transpose((1, 2, 0))  # rescale for matplotlib
    color_image[:, :, 0:1] = color_image[:, :, 0:1] * 100
    color_image[:, :, 1:3] = color_image[:, :, 1:3] * 255 - 128   
    color_image[:, :, 0:1] = np.clip(color_image[:, :, 0:1], 0, 100)
    #color_image[:, :, 1:3] = np.clip(color_image[:, :, 1:3], -85, 85)
    color_image = lab2rgb(color_image.astype(np.float64))
    color_image = np.clip(color_image, 0, 1)
    
    return color_image



class Trainer:
    def __init__(self):
        # dataloaders
        self.dataset = ColorizeData()
        nbEx = len(self.dataset)
        self.nbTrain = int(0.8*nbEx)
        self.nbVal = nbEx - self.nbTrain
        train_dataset, val_dataset = torch.utils.data.random_split(self.dataset, [self.nbTrain,  self.nbVal])

        self.train_dataloader = DataLoader(train_dataset, batch_size=32,
                        shuffle=True)

        self.val_dataloader = DataLoader(val_dataset, batch_size=8,
                        shuffle=True)
        # Model
        self.model = Net()
        # Loss function to use
        self.criterion = mse_loss #l1_loss
        self.optimizer = Adam(self.model.parameters(),  lr=1e-2)
        
        
        
        pass
        # Define hparams here or load them from a config file
    def train(self, nbEpoch = 100, printIm=False):
        print("starting the training")


        size = len(self.train_dataloader.dataset)

        batchSize = 64
        pbar = tqdm(range(nbEpoch))
        
        Loss = []
        
        for i in pbar:

            try:
                img_grey, input_ab, img_original = next(iter(self.train_dataloader))
            except:
                continue

            output_ab = self.model(img_grey)
            
            loss = self.criterion(input_ab, output_ab)


            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            Loss.append(loss.item())
            pbar.set_description("loss %s" % loss.item())
            
            self.example(train=True)
            

        plt.xlabel("epochs")
        plt.plot(Loss)
        plt.title("Loss")
        plt.show()
        
        torch.save(self.model, "./trainedModel")


            
    def saveParam(self, name):
        save(self.model.state_dict(), "./saved/" + name)
    
    def loadParam(self, name):
        checkpoint = torch.load("./saved/" + name)
        self.model.load_state_dict(checkpoint)
        
    def validate(self, dataloader=None, lossFunction=None):
        print("starting the validation")
        if dataloader==None:
            dataloader = self.val_dataloader
            
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        test_loss = 0
        errors = 0
        if lossFunction == None:
            lossFunction = self.criterion

        with torch.no_grad():
            for i in tqdm(range(num_batches)):
                try:
                    img_grey, input_ab, img_original = next(iter(self.train_dataloader))
                except:
                    errors += 1
                    continue
                
                output_ab = self.model(img_grey)
                loss = lossFunction(input_ab, output_ab)
                test_loss += loss.item()

        test_loss /= (size - errors)
        print(f"Test Error: Avg loss: {test_loss:>8f} \n")
        
        return test_loss

    def example(self, train=False, save_path=None, save_name=None):
        
        if train:
            loader = self.train_dataloader
        else:
            loader = self.val_dataloader
        
        try:
            input_gray, input_ab, target = next(iter(loader))
            output_ab = self.model(input_gray)
        except:
            return
        
        input_gray = input_gray[0]
        output_ab = output_ab[0]
        target = target[0].detach().numpy()

        output = to_rgb(input_gray, ab_input=output_ab.detach())
 
        input_gray = input_gray.numpy().transpose((1, 2, 0)).squeeze()
        
        if (save_path is not None) and (save_name is not None): 
            os.makedirs(save_path, exist_ok=True)
            plt.imsave(arr=input_gray, fname='{}{}_gray.jpg'.format(save_path, save_name), cmap='gray')
            plt.imsave(arr=target, fname='{}{}_target.jpg'.format(save_path, save_name))
            plt.imsave(arr=output, fname='{}{}_colorized.jpg'.format(save_path, save_name))
        
        
        plt.imshow(target)
        plt.title("orignial image")
        plt.show()
        
        get_histogram(target)
        
        plt.imshow(output)
        plt.title("result")
        plt.show()
        
        get_histogram(output)


if __name__ == '__main__':
    
    trainer = Trainer()
    
    trainer.train(printIm=True)
    #trainer.loadParam("1000epLAB")
    trainer.validate()
    trainer.example(train=False, save_path='./generated/', save_name='gen1')
    