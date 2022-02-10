import os
import codecs
import numpy as np
import torch
import datetime
import time

import csv
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import shutil
import os

#boader information of partitions
def load_fence(filepath):        
    buffer = []
    with open(filepath, 'rt', encoding='utf-8-sig') as vsvfile:
        reader = csv.reader(vsvfile)
        for row in reader:
            buffer = buffer + (list(map(float,row)))
    buffer = np.array(buffer).reshape(len(buffer) // 2 , 2)
    return buffer

fencedc = load_fence("/.../CNNTCN/Geolife/bjDistrict/dongcheng.txt")
fencexc = load_fence("/.../CNNTCN/Geolife/bjDistrict/xicheng.txt")
fencecy = load_fence("/.../CNNTCN/Geolife/bjDistrict/chaoyang.txt")
fenceft = load_fence("/.../CNNTCN/Geolife/bjDistrict/fengtai.txt")
fencesjs = load_fence("/.../CNNTCN/Geolife/bjDistrict/shijingshan.txt")
fencehd = load_fence("/.../CNNTCN/Geolife/bjDistrict/haidian.txt")
fencemtg = load_fence("/.../CNNTCN/Geolife/bjDistrict/mentougou.txt")
fencefs = load_fence("/.../CNNTCN/Geolife/bjDistrict/fangshan.txt")
fencetz = load_fence("/.../CNNTCN/Geolife/bjDistrict/tongzhou.txt")
fencesy = load_fence("/.../CNNTCN/Geolife/bjDistrict/shunyi.txt")
fencecp = load_fence("/.../CNNTCN/Geolife/bjDistrict/changping.txt")
fencedx = load_fence("/.../CNNTCN/Geolife/bjDistrict/daxing.txt")
fencehr = load_fence("/.../CNNTCN/Geolife/bjDistrict/huairou.txt")
fencepg = load_fence("/.../CNNTCN/Geolife/bjDistrict/pinggu.txt")
fencemy = load_fence("/.../CNNTCN/Geolife/bjDistrict/miyun.txt")
fenceyq = load_fence("/.../CNNTCN/Geolife/bjDistrict/yanqing.txt")

def IsIn(point, fence):
    if Polygon(fence).contains(point) == True:
        return 1
    else:
        return 0

#Partition
def func(T):
    num = {'dc':0, 'xc':0,'cy':0,'ft':0,
            'sjs':0,'hd':0,'mtg':0,'fs':0,
            'tz':0,'sy':0,'cp':0,'dx':0,
            'hr':0,'pg':0,'my':0,'yq':0,}
    for i in T:
        point = Point(float(i[1]),float(i[0]))
        num['dc'] += IsIn(point, fencedc)
        num['xc'] += IsIn(point, fencexc)
        num['cy'] += IsIn(point, fencecy)
        num['ft'] += IsIn(point, fenceft)
        num['sjs'] += IsIn(point, fencesjs)
        num['hd'] += IsIn(point, fencehd)
        num['mtg'] += IsIn(point, fencemtg)
        num['fs'] += IsIn(point, fencefs)
        num['tz'] += IsIn(point, fencetz)
        num['sy'] += IsIn(point, fencesy)
        num['cp'] += IsIn(point, fencecp)
        num['dx'] += IsIn(point, fencedx)
        num['hr'] += IsIn(point, fencehr)
        num['pg'] += IsIn(point, fencepg)
        num['my'] += IsIn(point, fencemy)
        num['yq'] += IsIn(point, fenceyq)

    maxv = max(num.values())
    if maxv == 0:
        return 0
    for key,value in num.items():
        if(value == maxv):
            if key=='dc' or key=='xc':
                return 3
            else:
                if key=='hd' or key=='cy' or key=='sjs' or key=='ft':
                    return 1
                else:
                    return 2

#get train set and test set
def data_generatorDis(batch_size):
    path_train = "/.../TCN/Geolife/data3/11/train" #11: City Center, 22: Suburb, 33: Urban area
    path_test = "/.../TCN/Geolife/data2/test"
    list1 = os.listdir(path_train)
    list2 = os.listdir(path_test)
    X_train = []
    Y_train = [] #labels
    X_test1 = []
    Y_test1 = [] #labels
    X_test2 = []
    Y_test2 = [] #labels
    X_test3 = []
    Y_test3 = [] #labels
    timefirst = datetime.datetime.strptime('2000-01-01 00:00:00', '%Y-%m-%d %H:%M:%S')
    #train set
    for i in list1:
        if int(i)<7:
            path1 = os.path.join(path_train,i)
            list11 = os.listdir(path1)
            for j in list11: 
                path11 = os.path.join(path1,j)
                f = open(path11, "r")
                line = f.readline()
                t = []
                while line:
                    a = line.strip('\n').split(',')
                    a[0] = float(a[0])
                    a[1] = float(a[1])
                    a[2] = (datetime.datetime.strptime(a[2], '%Y-%m-%d %H:%M:%S') - timefirst).seconds
                    t.append(a)
                    line = f.readline()
                f.close()
                if len(t) < 300: #trajectory length
                    continue
                else:
                    test = []
                    for k in range(0,300):
                        test.append(t[k])
                    X_train.append(test)
            
                y=[0,0,0,0,0,0,0]
                y[int(i)]=1
                Y_train.append(y)
        else:continue
        
    X_train = torch.tensor(X_train, dtype=torch.float32)
    Y_train = torch.tensor(Y_train, dtype=torch.long)

    datasets = torch.utils.data.TensorDataset(X_train,Y_train)
    train_loader = torch.utils.data.DataLoader(datasets, batch_size=batch_size)
    
    #test set
    for i in list2: 
            path1 = os.path.join(path_test,i) 
            list11 = os.listdir(path1) 
            for j in list11: 
                path11 = os.path.join(path1,j)
                f = open(path11, "r")
                line = f.readline()
                t = []
                while line:
                    a = line.strip('\n').split(',')
                    a[0] = float(a[0])
                    a[1] = float(a[1])
                    a[2] = (datetime.datetime.strptime(a[2], '%Y-%m-%d %H:%M:%S') - timefirst).seconds
                    t.append(a)
                    line = f.readline()
                f.close()
                if len(t) <= 300:
                    continue
                else:
                    test = []
                    for k in range(0,300):
                        test.append(t[k])
                    m = func(test)
                    if m==0: continue
                    else:
                        y=[0,0,0,0,0,0,0]
                        y[int(i)]=1
                        if m==1:
                            X_test1.append(test)
                            Y_test1.append(y)
                        if m==2:
                            X_test2.append(test)
                            Y_test2.append(y)
                        if m==3: 
                            X_test3.append(test)
                            Y_test3.append(y)
        else:continue

    X_test1 = torch.tensor(X_test1, dtype=torch.float32)
    Y_test1 = torch.tensor(Y_test1, dtype=torch.long)
    datasets1 = torch.utils.data.TensorDataset(X_test1,Y_test1)
    test_loader1 = torch.utils.data.DataLoader(datasets1, batch_size=batch_size)

    X_test2 = torch.tensor(X_test2, dtype=torch.float32)
    Y_test2 = torch.tensor(Y_test2, dtype=torch.long)
    datasets2 = torch.utils.data.TensorDataset(X_test2,Y_test2)
    test_loader2 = torch.utils.data.DataLoader(datasets2, batch_size=batch_size)

    X_test3 = torch.tensor(X_test3, dtype=torch.float32)
    Y_test3 = torch.tensor(Y_test3, dtype=torch.long)
    datasets3 = torch.utils.data.TensorDataset(X_test3,Y_test3)
    test_loader3 = torch.utils.data.DataLoader(datasets3, batch_size=batch_size)
    
    return train_loader, test_loader1, test_loader2, test_loader3

#train set and test sets of 3 partitions
t1,t2,t3,t4 = data_generatorDis(64)
