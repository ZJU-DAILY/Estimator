import matplotlib.pyplot as plt
import numpy as np
from math import sin,cos,tan,pi,radians,asin,sqrt,degrees,atan2
import datetime
import time
import os
import codecs
from sklearn import preprocessing
deltalon = 0.3
deltalat = 0.3 #0.3
#grid image consisted of h*l(w), grid cell consisted of a*a
h = 50
l = 50
a = 1

#Distance Computation
def geodistance(lng1,lat1,lng2,lat2):
    lng1, lat1, lng2, lat2 = map(radians, [float(lng1), float(lat1), float(lng2), float(lat2)]) # 经纬度转换成弧度
    dlon=lng2-lng1
    dlat=lat2-lat1
    a=sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    distance=2*asin(sqrt(a))*6371*1000
    distance=round(distance/1000,3)
    return distance

#Azimuth Computation
def getDegree(latA, lonA, latB, lonB):
    radLatA = radians(latA)
    radLonA = radians(lonA)
    radLatB = radians(latB)
    radLonB = radians(lonB)
    dLon = radLonB - radLonA
    y = sin(dLon) * cos(radLatB)
    x = cos(radLatA) * sin(radLatB) - sin(radLatA) * cos(radLatB) * cos(dLon)
    brng = degrees(atan2(y, x))
    brng = (brng + 360) % 360
    return brng

#Centroid Computation
def Centroid_calculate(T):
    n = len(T)
    cenlat = 0
    cenlon = 0
    for i in range(n):
        cenlat = cenlat + T[i][0]/n
        cenlon = cenlon + T[i][1]/n
    return cenlat, cenlon

#Trajectory Mapping
def trajectory_mapping(T):
    cenlat, cenlon = Centroid_calculate(T)
    minlon = 200
    minlat = 200
    for i in range(len(T)):
        if T[i][0]<minlat:
            minlat = T[i][0]
        if T[i][1]<minlon:
            minlon = T[i][1]
    offsetx = np.floor((h*a)/2) - (cenlat-minlat)*h*a/deltalat #纬度
    offsety = np.floor((l*a)/2) - (cenlon-minlon)*l*a/deltalon #经度
    return offsetx, offsety, minlat, minlon

#Pixels Computation
def Image_calculate(T):
    offsetx, offsety, minlat, minlon = trajectory_mapping(T)

    Im = [[[0,0,0,0] for col in range(h)] for row in range(l)]
    
    for k in range(len(T)):
        x = (T[k][0]-minlat)*h*a/deltalat+offsetx
        y = (T[k][1]-minlon)*l*a/deltalon+offsety
        axisx = int(np.floor(x/a)) #coordinates in grid
        axisy = int(np.floor(y/a))
        if axisx>=h or axisx<0 or axisy>=l or axisy<0: continue
        lastid = k
        if Im[axisx][axisy][0]==0: #start point
            Im[axisx][axisy][0] = Im[axisx][axisy][0] + 1
            Im[axisx][axisy][1] = k
            Im[axisx][axisy][2] = k
            Im[axisx][axisy][3] = 0
        else:
            Im[axisx][axisy][0] = Im[axisx][axisy][0] + 1
            Im[axisx][axisy][3] = Im[axisx][axisy][3] + geodistance(T[k][1], T[k][0], T[lastid][1], T[lastid][0])
            Im[axisx][axisy][2] = k #end point

    #pixels
    total = []
    for i in range(h):
        for j in range(l):
            dist = Im[i][j][3]
            t1 = 0
            t2 = 0
            t1 = datetime.datetime.strptime(T[Im[i][j][1]][2], '%Y-%m-%d %H:%M:%S') #起点
            t2 = datetime.datetime.strptime(T[Im[i][j][2]][2], '%Y-%m-%d %H:%M:%S') #终点
            staytime = (t2 - t1).seconds
            direction = getDegree(T[Im[i][j][1]][0], T[Im[i][j][1]][1], T[Im[i][j][2]][0], T[Im[i][j][2]][1])

            if staytime == 0:
                total.append([0,0,0])
                continue
            ttt = []
            ttt.append(direction)
            ttt.append(dist/(staytime))
            ttt.append(staytime)
            total.append(ttt)
    #normalization
    min_max_scaler = preprocessing.MinMaxScaler() 
    X_minMax = min_max_scaler.fit_transform(total)
    
    k = 0
    Image = []
    for i in range(h):
        test1 = []
        for j in range(l):
            t = []
            t.append(X_minMax[k][0])
            t.append(X_minMax[k][1])
            t.append(X_minMax[k][2])
            test1.append(t)
            k = k + 1
        Image.append(test1)

    return Image

#Remove the white boader of images
def corp_margin(img):
    img2=img.sum(axis=2)
    (row,col)=img2.shape
    row_top=0
    raw_down=0
    col_top=0
    col_down=0
    for r in range(0,row):
        if img2.sum(axis=1)[r]<700*col:
            row_top=r
            break
    for r in range(row-1,0,-1):
        if img2.sum(axis=1)[r]<700*col:
            raw_down=r
            break
    for c in range(0,col):
        if img2.sum(axis=0)[c]<700*row:
            col_top=c
            break
    for c in range(col-1,0,-1):
        if img2.sum(axis=0)[c]<700*row:
            col_down=c
            break
    new_img=img[row_top:raw_down+1,col_top:col_down+1,0:3]
    return new_img

#get mapped trajectory
def Paint(Image, path):
    savepath = "/.../test.jpg"
    plt.axis('off')  
    plt.imshow(Image, interpolation='nearest')
    plt.savefig(savepath)
    im = plt.imread(savepath)
    img_re = corp_margin(im)
    plt.imsave(path, img_re)

# path1: direction of trajectories
# path2: direction to save mapped trajectories
def func(path1, path2):
    k = 0
    list1 = os.listdir(path1)
    for i in list1:
        p1 = os.path.join(path1,i)
        list2 = os.listdir(p1)
        for j in list2:
            print(os.path.join(p1, j))
            f = open(os.path.join(p1, j), "r")
            line = f.readline()
            ls = []
            while line:
                b = line.split(',')
                b[0] = float(b[0])
                b[1] = float(b[1])
                b[2] = b[2].strip('\n')
                ls.append(b)
                line = f.readline()
            f.close()
            Image = Image_calculate(ls)
            Paint(Image, path2+"\\"+str(i)+"\\IMAGE"+str(j)+".jpg")
            k = k + 1

func("/.../Geolife/data1/train", "/.../CNN/Image/train")
func("/.../Geolife/data1/test", "/.../CNN/Image/test")