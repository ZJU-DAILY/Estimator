import os
import codecs
import numpy as np
import torch
import datetime
import time

#get train set and test set
def data_generator(batch_size):
    path_train = "/.../CNNTCN/Geolife/data2/train"
    path_test = "/.../CNNTCN/Geolife/data2/test"

    list1 = os.listdir(path_train)
    list2 = os.listdir(path_test)
    X_train = []
    Y_train = [] #labels
    X_test = []
    Y_test = [] #labels
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
            
                if len(t) <= 300:continue
                else:
                    test = []
                    for k in range(0,300):
                        test.append(t[k])
                    X_train.append(test)

                y = [0,0,0,0,0,0,0]
                y[int(i)] = 1
                Y_train.append(y)
        else:continue
    X_train = torch.tensor(X_train, dtype=torch.float32)
    Y_train = torch.tensor(Y_train, dtype=torch.long)

    datasets = torch.utils.data.TensorDataset(X_train,Y_train)
    train_loader = torch.utils.data.DataLoader(datasets, batch_size=batch_size)
    
    #test set
    for i in list1:
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
                    X_test.append(test)

                y = [0,0,0,0,0,0,0]
                y[int(i)] = 1
                Y_test.append(y)
        else:continue

    X_test = torch.tensor(X_test, dtype=torch.float32)
    Y_test = torch.tensor(Y_test, dtype=torch.long)

    datasets = torch.utils.data.TensorDataset(X_test,Y_test)
    test_loader = torch.utils.data.DataLoader(datasets, batch_size=batch_size)
    
    return train_loader, test_loader
