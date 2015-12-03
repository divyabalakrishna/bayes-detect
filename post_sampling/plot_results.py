from numpy import *
import copy
from copy import deepcopy
import scipy
import os,sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MeanShift, estimate_bandwidth
from ConfigParser import SafeConfigParser
from itertools import cycle
from scipy.signal import argrelextrema
from datetime import *
from math import *

import seaborn as sns #makes the plots look pretty

from common import *

def filterCluster(XX,YY,x,y,a,r,l,output_folder):
    X = copy.deepcopy(x)
    Y = copy.deepcopy(y)
    A = copy.deepcopy(a)
    R = copy.deepcopy(r)
    L = copy.deepcopy(l)
    xx = []
    yy = []
    aa = []
    rr = []
    ll = []
    for i in range(len(XX)):
        index = where(X==XX[i])
        xx.append(X[index][0])
        yy.append(Y[index][0])
        aa.append(A[index][0])
        rr.append(R[index][0])
        ll.append(L[index][0])
    temp = zeros((len(xx),5))
    temp[:,0] = xx
    temp[:,1] = yy
    temp[:,2] = aa
    temp[:,3] = rr
    temp[:,4] = ll
    savetxt(output_folder + "/cluster.txt", temp,fmt='%.6f')
    X,Y,A,R,L = loadtxt(output_folder + "/cluster.txt", unpack=True)
    return X,Y,A,R,L

def dbscan(x,y):
    YY=zeros((len(x),2))
    YY[:,0]=x
    YY[:,1]=y
    N = len(YY[:,0])
    length = sorted(YY[:,0])[-1] - sorted(YY[:,0])[0]
    breath = sorted(YY[:,1])[-1] - sorted(YY[:,1])[0]
    eps = 4*(sqrt(length*breath/N))
    db = DBSCAN(eps = eps).fit(YY)
    core_samples_mask = zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    unique_labels = set(labels)
    colors = plt.cm.jet(linspace(0, 1, len(unique_labels)))
    clusters = [YY[labels == i] for i in xrange(n_clusters_)]
    return clusters

def is_hit(data1, data2):
    distance = ((data1[0] - data2[0])**2 + (data1[1] - data2[1])**2)**0.5
    if distance < (data1[3] + data2[3]):
        return 1
    else:
        return 0

def post_run(output_folder,prefix):
    originalData = load(output_folder +"/" + prefix + "_srcs.npy")
    originalData = originalData[originalData[:,1].argsort()]
    finalData = loadtxt(output_folder +"/finalData.txt")
    finalData = finalData[finalData[:,1].argsort()]
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    i = 0
    while i < len(finalData):
        j = 0
        while j < len(originalData):
            a = is_hit(finalData[i],originalData[j])
            if(a == 1):
                tp = tp + 1
                finalData = delete(finalData,i,0)
                originalData = delete(originalData,j,0)
                i = 0
                j = 0
                break
            j = j + 1
        if j == len(originalData):
            i = i + 1
    fp = len(finalData)
    print "TP: ",tp,"FP: ",fp,"Undetected: ",len(originalData)
    return tp, fp, len(originalData)

def getMinMax(x,y,l,width,height):
    w, xmask, xm, Lmx = binned_max(x, l, 0, width, 400)
    smoothed_x = smooth(Lmx[xmask])
    maxesX = compute_maxes(xm[xmask], smoothed_x)
    minsX = compute_mins(xm[xmask], smoothed_x)

    w, ymask, ym, Lmy = binned_max(y, l, 0, height, 400)
    smoothed_y = smooth(Lmy[ymask])
    maxesY = compute_maxes(ym[ymask], smoothed_y)
    minsY = compute_mins(ym[ymask], smoothed_y)
    return maxesX,maxesY,minsX,minsY

def sliceValues(x,y,a,r,l,min,max,axis1):
    ZZ=zeros((len(x),5))
    ZZ[:,0]=x
    ZZ[:,1]=y
    ZZ[:,2]=a
    ZZ[:,3]=r
    ZZ[:,4]=l
    i = 0
    while i < len(ZZ):
        if axis1 == 'x':
            if(ZZ[i][0] < min or ZZ[i][0] > max):
                ZZ = delete(ZZ, i, axis=0)
            else:
                i = i+1 
        else:
            if(ZZ[i][1] < min or ZZ[i][1] > max):
                ZZ = delete(ZZ, i, axis=0)
            else:
                i = i+1
    x=ZZ[:,0]
    y=ZZ[:,1]
    a=ZZ[:,2]
    r=ZZ[:,3]
    l=ZZ[:,4]
    return x,y,a,r,l

def correctData(maxPoints, minPoints):
    list1 = sorted(maxPoints + minPoints)
    i = 0
    while i < len(list1):
        if i == 0:
            if list1[i] in minPoints: prev = 'min'
            elif list1[i] in maxPoints: prev = 'max'
        else:
            if list1[i] in minPoints: cur = 'min'
            elif list1[i] in maxPoints: cur = 'max'
            if prev == cur:
                new = (list1[i-1] + list1[i])/2
                if prev == 'min':
                    minPoints.remove(list1[i])
                    minPoints.remove(list1[i-1])
                    minPoints.append(new)
                    minPoints = sorted(minPoints)
                if prev == 'max':
                    maxPoints.remove(list1[i])
                    maxPoints.remove(list1[i-1])
                    maxPoints.append(new)
                    maxPoints = sorted(maxPoints)
                list1.remove(list1[i])
                list1.remove(list1[i-1])
                list1.append(new)
                list1 = sorted(list1)
                i = i - 1
            prev = cur
        i = i + 1
    return list1

def getObjects(x1,y1,a1,r1,l1,objects,width,height):
    maxesX,maxesY,minsX,minsY = getMinMax(x1,y1,l1,width,height)
    correctData(maxesX,minsX)
    correctData(maxesY,minsY)
    if (len(maxesX) == 1) and (len(maxesY) == 0):
        index = argmax(l1)
        objects.append([x1[index],y1[index],a1[index],r1[index],l1[index]])
        return objects
    if (len(maxesX) == 0) and (len(maxesY) == 1):
        index = argmax(l1)
        objects.append([x1[index],y1[index],a1[index],r1[index],l1[index]])
        return objects
    if (len(maxesX) == 1) and (len(maxesY) == 1):
        index = argmax(l1)
        objects.append([x1[index],y1[index],a1[index],r1[index],l1[index]])
        return objects
    elif len(maxesX) > 1:
        if len(minsX) >= len(maxesX)-1:
            mx = []
            mx.append(min(x1))
            for i in minsX:
                mx.append(i)
            mx.append(max(x1))
            for i in range(len(maxesX)):
                index0 = 0
                index1 = 1
                while not(maxesX[i] > mx[index0] and maxesX[i] < mx[index1]) and index1 < len(mx)-1:
                    index0 = index0+1
                    index1 = index0+1
                x2,y2,a2,r2,l2 = sliceValues(x1,y1,a1,r1,l1,mx[index0],mx[index1],'x')
                objects = getObjects(x2,y2,a2,r2,l2,objects,width,height)
            return objects
        else:
            index = argmax(l1)
            objects.append([x1[index],y1[index],a1[index],r1[index],l1[index]])
            return objects
    elif len(maxesY) > 1:
        if len(minsY) >= len(maxesY)-1:
            my = []
            my.append(min(y1))
            for i in minsY:
                my.append(i)
            my.append(max(y1))
            for i in range(len(maxesY)):
                index0 = i
                index1 = i+1
                while not(maxesY[i] > my[index0] and maxesY[i] < my[index1]) and index1 < len(my)-1:
                    index0 = index0+1
                    index1 = index0+1
                x2,y2,a2,r2,l2 = sliceValues(x1,y1,a1,r1,l1,my[index0],my[index1],'y')
                objects = getObjects(x2,y2,a2,r2,l2,objects,width,height)
            return objects
        else:
            index = argmax(l1)
            objects.append([x1[index],y1[index],a1[index],r1[index],l1[index]])
            return objects
    return objects

def run(configfile):
    try:
        os.mkdir('plots')
    except:
        pass

    parser = SafeConfigParser()
    parser.read(configfile)

    width = int(parser.get("Sampling", "width"))
    height = int(parser.get("Sampling", "height"))

    amp_min = float(parser.get("Sampling", "amp_min"))
    amp_max = float(parser.get("Sampling", "amp_max"))

    rad_min = float(parser.get("Sampling", "rad_min"))
    rad_max = float(parser.get("Sampling", "rad_max"))

    prefix = parser.get("Misc", "prefix")
    location = parser.get("Misc", "location")
    output_folder = location + "/" + prefix 
    
    #output parameters
    output_filename = prefix + "_" + parser.get("Output", "output_filename")
    
    x,y,a,r,l = loadtxt(output_folder + "/active_points.txt", unpack=True)
    #x,y,a,r,l = loadtxt(output_folder + "/" + output_filename, unpack=True)

    clusters = dbscan(x,y)
    print "no of clusters:",len(clusters)
    objects = []

    objects = []
    for i in range(len(clusters)):
        x1,y1,a1,r1,l1 = filterCluster(clusters[i][:,0],clusters[i][:,1],x,y,a,r,l,"/") 
        objects = getObjects(x1,y1,a1,r1,l1,objects,width,height)
    cleanlist = []
    [cleanlist.append(x) for x in objects if x not in cleanlist]
    print len(cleanlist)
    fig= plt.figure(1,figsize=(10,10), dpi=100)
    proj = fig.add_subplot(111)
    for i in range(len(cleanlist)):
        circle =  plt.Circle((cleanlist[i][0],cleanlist[i][1]),cleanlist[i][3],color = 'b',facecolor='none', alpha=0.3)
        fig = plt.gcf()
        fig.gca().add_artist(circle)
    
    originalData = np.load(output_folder +"/" + prefix + "_srcs.npy")

    
    for i in range(len(originalData)):
        circle =  plt.Circle((originalData[i][0],originalData[i][1]),originalData[i][3],color = 'r',facecolor='none', alpha=0.3)
        fig = plt.gcf()
        fig.gca().add_artist(circle)
        
    coordsX = []
    coordsY = []
    
    for row in cleanlist:
        coordsX.append(row[0])
        coordsY.append(row[1])
    proj.plot(coordsX, coordsY, 'o', markerfacecolor="b", markersize=3)
    proj.set_xlim(0,width)
    proj.set_ylim(0,height)
    
    proj.plot(originalData[:,0], originalData[:,1], 'o', markerfacecolor="r", markersize=3)
    plt.savefig(output_folder + "/clusters_active_points.png", bbox_inches="tight")
    plt.show()
    
    savetxt(output_folder +"/finalData.txt", cleanlist,fmt='%.6f')
    tp, fp, ud = post_run(output_folder,prefix)
    return tp,fp,ud

