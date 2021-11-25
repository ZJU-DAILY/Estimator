import torch
from torch import nn, optim
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim  
#import matplotlib.pyplot as plt
import numpy as np
import sys, os
sys.path.append('/...')
from CNN.res50 import ResNet50
import matplotlib.pyplot as plt
from math import sin,cos,tan,pi,radians,asin,sqrt,degrees,atan2
import datetime
import time
import os
import codecs
from sklearn import preprocessing
from PIL import Image

classes = ('0','1', '2', '3', '4', '5', '6')
deltalon = 0.3
deltalat = 0.3 
#grid image consisted of h*l (h*w in the paper),grid cell is consisted of a*a
h = 50
l = 50
a = 1
device = torch.device('cuda')

def loaddata(path,batchsz):
    trainset = torchvision.datasets.ImageFolder(path,
                                                transform=transforms.Compose([
                                                    transforms.Resize((32, 32)),

                                                    transforms.CenterCrop(32),
                                                    transforms.ToTensor()])
                                                )
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchsz,
                                              shuffle=True, num_workers=2)
    return trainloader

def main():
    batchsz = 64

    path1 = "/.../CNN/Image/Geolife/train" # direction of train set of Image
    Image_train = loaddata(path1, batchsz)

    x, label = iter(Image_train).next()
    print('x:', x.shape, 'label:', label.shape)

    model = ResNet50().to(device)

    criteon = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(80):
        model.train()
        for batchidx, (x, label) in enumerate(Image_train):
            x, label = x.to(device), label.to(device)

            logits = model(x)
            loss = criteon(logits, label)

            # backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(epoch, 'loss:', loss.item())
        model.eval()

        
    print('Finished Training')
    # save the parameters of trained network
    torch.save(model, 'model.pkl') 
    torch.save(model.state_dict(), 'model_params.pkl') 
    
def test():
    batchsz = 64

    path2 = "/.../CNN/Image/Geolife/test"
    Image_test = loaddata(path2,batchsz)
    
    start_time = time.time()

    model = torch.load('model.pkl')
    model = model.to(device)

    with torch.no_grad():
        total_num = 0
        total_correct = 0

        for x, label in Image_test:
            x, label = x.to(device), label.to(device)

            logits = model(x)
            
            pred = logits.argmax(dim=1)
            correct = torch.eq(pred,label).float().sum().item()
            total_correct += correct
            total_num += x.size(0)

        acc = total_correct/total_num
        print('test acc:{}/{}'.format(total_correct,total_num),acc)
        
    end_time = time.time()
    print("test time:{:.2f}".format(end_time-start_time))

if __name__ == '__main__':
    #main()
    test()
