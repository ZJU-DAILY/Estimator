import torch
from torch.autograd import Variable
import torch.nn.functional as F
import sys, os
sys.path.append("/.../")
from CNNTCN.Geolife.utilsT import data_generatorDis
from CNNTCN.Geolife.utilsC import data_generator
from CNNTCN.Geolife.model import TCN
from CNN.trajectory_mapping import geodistance, getDegree, Centroid_calculate, trajectory_mapping, Image_calculate, corp_margin
import numpy as np
import argparse
from torch import nn, optim
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import DataLoader
from torch.autograd import Variable 
from CNN.res50 import ResNet50
import matplotlib.pyplot as plt
from math import sin,cos,tan,pi,radians,asin,sqrt,degrees,atan2
import datetime
import time
import os
import codecs
from sklearn import preprocessing
from PIL import Image as Im
import multiprocessing
import socket
import json

start_time = time.time()

classes = ('0','1', '2', '3', '4', '5', '6')
deltalon = 0.3
deltalat = 0.3 #0.3
h = 50
l = 50
a = 1
device = torch.device('cuda')
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose

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

#predict the class of image
def predict(img): 
    net=torch.load('cnnmodel.pkl')
    net=net.to(device)
    torch.no_grad()
    
    transform = transforms.Compose([
        transforms.Resize(256,interpolation=2),
        transforms.ToTensor()
        ])
    img=transform(img).unsqueeze(0)
    img_ = img.to(device)
    outputs = net(img_)
    _, predicted = torch.max(outputs, 1)
    return outputs

def cnnTest(list1):
    savepath = "/.../CNN/test.jpg"
    Image = Image_calculate(list1)
    plt.axis('off')  #去掉坐标轴
    plt.imshow(Image, interpolation='nearest')
    plt.savefig(savepath)
    img = plt.imread(savepath)
    new_img = corp_margin(img)
    plt.imsave(savepath,new_img)
    plt.imshow(new_img)
    new_img = Im.open(savepath)
    return predict(new_img)

parser = argparse.ArgumentParser(description='Sequence Modeling - (Permuted) Sequential TCN')
parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='batch size (default: 64)')
parser.add_argument('--cuda', action='store_false',
                    help='use CUDA (default: True)')
parser.add_argument('--dropout', type=float, default=0.05,
                    help='dropout applied to layers (default: 0.05)')
parser.add_argument('--clip', type=float, default=-1,
                    help='gradient clip, -1 means no clip (default: -1)')
parser.add_argument('--epochs', type=int, default=20,
                    help='upper epoch limit (default: 20)')
parser.add_argument('--ksize', type=int, default=7,
                    help='kernel size (default: 7)')
parser.add_argument('--levels', type=int, default=8,
                    help='# of levels (default: 8)')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='report interval (default: 100')
parser.add_argument('--lr', type=float, default=2e-3,
                    help='initial learning rate (default: 2e-3)')
parser.add_argument('--optim', type=str, default='Adam',
                    help='optimizer to use (default: Adam)')
parser.add_argument('--nhid', type=int, default=25,
                    help='number of hidden units per layer (default: 25)')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed (default: 1111)')
parser.add_argument('--permute', action='store_true',
                    help='use permuted TCN (default: false)')
args = parser.parse_args()

torch.manual_seed(args.seed)
if torch.cuda.is_available(): 
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

batch_size = args.batch_size
n_classes = 7
input_channels = 3
seq_length = int(300*3 / input_channels)
epochs = args.epochs
steps = 0

print(args)
train_loaderC, test_loader = data_generator(batch_size) #get test set of CNN
train_loader, test_loader1, test_loader2, test_loader3 = data_generatorDis(batch_size) #get train set and test set of TCN

permute = torch.Tensor(np.random.permutation(1024).astype(np.float64)).long()
channel_sizes = [args.nhid] * args.levels
kernel_size = args.ksize
model = TCN(input_channels, n_classes, channel_sizes, kernel_size=kernel_size, dropout=args.dropout)

if args.cuda:
    model.cuda()
    permute = permute.cuda()

lr = args.lr #learning rate
optimizer = getattr(optim, args.optim)(model.parameters(), lr=lr) #返回对象属性值

#TCN training
def train(ep):
    global steps
    train_loss = 0
    model.train()
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(train_loader):

            if args.cuda: data, target = data.cuda(), target.cuda()
            data = data.view(-1, input_channels, seq_length)
            if args.permute:
                data = data[:, :, permute]
            data, target = Variable(data), Variable(target)
            print("data:",data.size())
            print("target:",target.size())
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss = loss.requires_grad_()
            loss.backward()
            if args.clip > 0: 
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            train_loss += loss
            steps += seq_length

            print("output size:",output.shape)

            if batch_idx > 0 and batch_idx % args.log_interval == 0: 
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tSteps: {}'.format(
                    ep, batch_idx * batch_size, len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), train_loss.item()/args.log_interval, steps))
                train_loss = 0
        l = len(train_loader.dataset)
        print("l:",l)
 
        print("finish training")
        torch.save(model,'Urbanmodel.pkl')
        torch.save(model.state_dict(),'Urbanmodel_params.pkl')

from sklearn.metrics import classification_report,f1_score
from sklearn.feature_selection import VarianceThreshold
import pandas as pd

#TCN testing (serial)
def test():
    net1 = torch.load('Urbanmodel20.pkl')
    net1 = net1.to(device)
    net2 = torch.load('Suburbmodel20.pkl')
    net2 = net2.to(device)
    net3 = torch.load('Urban2model20.pkl')
    net3 = net3.to(device)

    test_loss = 0
    correct = 0
    TP = 0

    labe = []
    da = []

    with torch.no_grad():
        for data, target in test_loader1:
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data = data.view(-1, input_channels, seq_length)
            if args.permute:
                data = data[:, :, permute]
            with torch.no_grad():
                data, target = Variable(data), Variable(target)
            output = net1(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
            p = pred.eq(target.data.view_as(pred))
            TP += ((p==True)&(target.data==1)).cpu().sum()
            
        for data, target in test_loader2:
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data = data.view(-1, input_channels, seq_length)
            if args.permute:
                data = data[:, :, permute]
            with torch.no_grad():
                data, target = Variable(data), Variable(target)
            output = net2(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
            p = pred.eq(target.data.view_as(pred))
            TP += ((p==True)&(target.data==1)).cpu().sum()

        for data, target in test_loader3:
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data = data.view(-1, input_channels, seq_length)
            if args.permute:
                data = data[:, :, permute]
            with torch.no_grad():
                data, target = Variable(data), Variable(target)
            output = net2(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
            p = pred.eq(target.data.view_as(pred))
            TP += ((p==True)&(target.data==1)).cpu().sum()

        l = len(test_loader1.dataset)+len(test_loader2.dataset)+len(test_loader3.dataset)
        test_loss /= l
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%), F1:({:.2f})\n'.format(test_loss, correct, l, 100.* correct / l, 2/(2+(l-correct)/FP)))
        return test_loss

# TCN testing (parallel)
def test1(test_loader, net):
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data,target in test_loader:
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data = data.view(-1,input_channels,seq_length)
            if args.permute:
                data = data[:,:,permute]
            data, target = Variable(data,volatile=True), Variable(target)
            output = net(data)
            test_loss += F.nll_loss(output,target,size_average=False).item()
            pred = output.data.max(1,keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
            p = pred.eq(target.data.view_as(pred))
            TP += ((p==True)&(target.data==1)).cpu().sum()
        print('\nTest set: Average loss:{:.4f}, Accuracy:{}/{}({:.2f}%), F1:{:.2f}\n'.format(test_loss/len(test_loader.dataset),correct,len(test_loader.dataset),100.*correct/len(test_loader.dataset), 2/(2+(len(test_loader.dataset)-correct)/FP)))
        return test_loss, correct, TP

#parallel
def func(net,test_loader):
    ip_port = ("127.0.0.1", 9001)
    sk = socket.socket()
    sk.connect(ip_port)
    loss, correct, TP = test1(test_loader,net)
    t = []
    t.append(loss)
    t.append(int(correct))
    t.append(int(TP))
    t = json.dumps(t)
    sk.send(bytes(t.encode('utf-8')))
    sk.close()

if __name__ == "__main__":
    start_time = time.time()
    '''
    for epoch in range(1, epochs+1):
        train(epoch)
    '''
    mid_time = time.time()
    #print("training time:{:.2f}".format(mid_time-start_time))


    cnnTest(text_loader)
    ctx = multiprocessing.get_context("spawn")
    loss = 0
    correct = 0
    TP = 0
    ip_port = ("127.0.0.1", 9001)
    sk = socket.socket()
    sk.bind(ip_port)
    sk.listen(3)       
    p1 = ctx.Process(target=func,args=(net1,test_loader1))
    p1.start()
    p2 = ctx.Process(target=func,args=(net2,test_loader2))
    p2.start()
    p3 = ctx.Process(target=func,args=(net3,test_loader3))
    p3.start()
    for i in range(3):
        conn, addr = sk.accept()
        data = conn.recv(1024)
        data = json.loads(data)
        print(data)
        loss += data[0]
        correct += data[1]
        conn.close()
     p1.join()
     p2.join()
     p3.join()
     sk.close()
     print('\nTest set:Total Average loss:{:.4f},Total Accuracy:{}/{}({:0.2f}%)\n'.format(loss/lens,correct,lens,100*correct/lens))
     end_time = time.time()
     print("testing time:%d" %(end_time-mid_time))

