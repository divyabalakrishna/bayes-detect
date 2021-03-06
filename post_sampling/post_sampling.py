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


def filterNoise(XX,YY,x,y,a,r,l):
    X = copy.deepcopy(x)
    Y = copy.deepcopy(y)
    A = copy.deepcopy(a)
    R = copy.deepcopy(r)
    L = copy.deepcopy(l)
    for i in range(len(XX)):
        index = where(X==XX[i])
        X = delete(X,index)
        Y = delete(Y,index)
        A = delete(A,index)
        R = delete(R,index)
        L = delete(L,index)
    return X,Y,A,R,L

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

def dbscan(XX,name,x,y,a,r,l):
    #DBSCAN
    X = []
    Y = []
    L = []
    A = []
    R = []
    N = len(XX[:,0])
    length = sorted(XX[:,0])[-1] - sorted(XX[:,0])[0]
    breath = sorted(XX[:,1])[-1] - sorted(XX[:,1])[0]
    eps = 2*(sqrt(length*breath/N))
    db = DBSCAN(eps=eps).fit(XX)
    print eps,length,breath
    core_samples_mask = zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_


    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    
    unique_labels = set(labels)
    colors = plt.cm.jet(linspace(0, 1, len(unique_labels)))
    clusters = [XX[labels == i] for i in xrange(n_clusters_)]
    
    plt.figure()
    centers = []
    for k, col in zip(unique_labels, colors):
        class_member_mask = (labels == k)
        xy = XX[class_member_mask & core_samples_mask]
        if k == -1:
            # Black used for noise.
            col = 'k'
            #X,Y,A,R,L = filterNoise(xy[:,0],xy[:,1],x,y,a,r,l)
        #plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,markeredgecolor='k', markersize=5)
        mx = mean(xy[:,0])
        my = mean(xy[:,1])
        centers.append([mx,my])
    return clusters,plt,centers,X,Y,A,R,L

def random_color():
    return plt.cm.gist_ncar(random.random())

def plot_segments(ax, locs, vals, min_vals, max_vals):
    """
    plots each segment with a different color
    where a segment should contain one peak
    """
    intervals = compute_intervals(min_vals, max_vals)
    intervals = intervals.astype("int")
    for x,y in intervals:
        if(x == y):
            x = 0
        lower_mask = locs > x
        upper_mask = locs < y
        mask = logical_and(lower_mask, upper_mask)
        ax.plot(locs[mask], vals[mask], color=random_color())
        #color is chosen randomly, so sometimes it makes a bad selection

def make_plot(filename, x, y, a, r, l, width, height, prefix, output_folder):
    #first plot of parameter vs L
    print "1"
    fig=plt.figure(figsize=(10,8))
    ax1=fig.add_subplot(2,2,1)

    ax1.scatter(x,y,s=3,marker='.')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_title('all active points')
    ax1.set_xlim(0,width)
    ax1.set_ylim(0,height)

    w, xmask, xm, Lmx = binned_max(x, l, 0, width, 600)

    print "2"
    ax2=fig.add_subplot(2,2,3)
    ax2.plot(x[w],l[w],'k,')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Likelihood')
    #ax2.plot(xm[xmask],Lmx[xmask],'r-')
    smoothed_x = smooth(Lmx[xmask])
    #ax2.plot(xm[xmask], smoothed_x, 'g-')
    mins = compute_mins(xm[xmask], smoothed_x)
    maxes = compute_maxes(xm[xmask], smoothed_x)
    if len(mins) != 0 and len(maxes) != 0:
        plot_segments(ax2, xm[xmask], smoothed_x, mins, maxes)
    ax2.set_title('X vs Likelhood')

    w, ymask, ym, Lmy = binned_max(y, l, 0, height, 600)

    print "3"
    ax4=fig.add_subplot(2,2,4)
    ax4.plot(y[w],l[w],'k,')
    ax4.set_xlim(0, width)
    ax4.set_xlabel('Y')
    ax4.set_ylabel('Likelihood')
    #ax4.plot(ym[ymask],Lmy[ymask],'r-')
    smoothed_y = smooth(Lmy[ymask])
    #ax4.plot(ym[ymask], smoothed_y, 'g-')
 
    mins = compute_mins(ym[ymask], smoothed_y)
    maxes = compute_maxes(ym[ymask], smoothed_y)
    if len(mins) != 0 and len(maxes) != 0:
        plot_segments(ax4, ym[ymask], smoothed_y, mins, maxes)

    ax4.set_title('Y vs Likelhood')
   
    print "4"
    ax5 = fig.add_subplot(2,2,2)
    data = load(output_folder + "/" + prefix + "_clean.npy")
    ax5.imshow(flipud(data),extent=[0,width,0,height])
    ax5.set_title('Original image ')

    print "save"

    plt.savefig(output_folder + "/"+filename+".png", bbox_inches="tight")
    
    #second plot of 3d parameters (x,y) vs L
    fig= plt.figure()

    proj = fig.add_subplot(111, projection='3d')
    proj.scatter(x[w],y[w],l[w],s=3,marker='.')
    proj.set_xlim(0,width)
    proj.set_ylim(0,height)
    proj.set_xlabel('X')
    proj.set_ylabel('Y')
    proj.set_zlabel('Likelihood')
    #proj.set_title('Posteriors in 3D after cut')
    plt.savefig(output_folder + "/3d_active.png", bbox_inches="tight")
    return w

def k_means(X, n_clusters):
    k_means = KMeans(n_clusters = n_clusters)
    k_means.fit(X)
    k_means_labels = k_means.labels_
    k_means_cluster_centers = k_means.cluster_centers_
    k_means_labels_unique = np.unique(k_means_labels)
    #print "cluster_count",len(k_means_cluster_centers)
    
    colors = plt.cm.Spectral(np.linspace(0, 1, len(k_means_labels_unique)))
    clusters = []
    for k, col in zip(range(len(k_means_cluster_centers)), colors):
        my_members = (k_means_labels == k)
        temp = zeros((len(X[my_members, 0]),2))
        temp[:,0] = X[my_members, 0]
        temp[:,1] = X[my_members, 1]
        cluster_center = k_means_cluster_centers[k]
        clusters.append(temp)
    return k_means_cluster_centers,clusters

def calculateRadius(x1,y1,a1,r1,l1):
    n = int(math.ceil((len(l1)*0.1)))
    temp = zeros((n,5))
    m = l1[0]
    r = r1[0]
    x = x1[0]
    y = y1[0]
    a = a1[0]
    for i in range(len(l1)):
        if(l1[i] > m):
            m = l1[i]
            a = a1[i]
            r = r1[i]
            x = x1[i]
            y = y1[i]
    return r,x,y,a,m

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
    #print finalData
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

    XX=zeros((len(x),2))
    XX[:,0]=x
    XX[:,1]=y
    clusters,plt,c,X,Y,A,R,L = dbscan(XX,"clusters_active_points",x,y,a,r,l)
    coordsX = []
    coordsY = []
    coordsR = []
    coordsA = []
    coordsL = []
    for i in range(len(clusters)):
        x1,y1,a1,r1,l1 = filterCluster(clusters[i][:,0],clusters[i][:,1],x,y,a,r,l,output_folder) 
        XX=zeros((len(x1),2))
        XX[:,0]=x1
        XX[:,1]=y1
        w, xmask, xm, Lmx = binned_max(x1, l1, 0, width, 200)
        smoothed_x = smooth(Lmx[xmask])
        maxes = compute_maxes(xm[xmask], smoothed_x)
        maxX = len(maxes)
        w, ymask, ym, Lmy = binned_max(y1, l1, 0, height, 200)
        smoothed_y = smooth(Lmy[ymask])
        maxes = compute_maxes(ym[ymask], smoothed_y)
        maxY = len(maxes)
        m = max(maxX,maxY)
        
        if m != 0:
            centers,kmeans_clusters = k_means(XX,m)
            colors = plt.cm.Spectral(np.linspace(0, 1, len(clusters)))
            for i, col in zip(range(len(centers)), colors):
                x2,y2,a2,r2,l2 = filterCluster(kmeans_clusters[i][:,0],kmeans_clusters[i][:,1],x,y,a,r,l,output_folder)
                coordsX.append(centers[i][0])
                coordsY.append(centers[i][1])
                #plt.plot(kmeans_clusters[i][:,0],kmeans_clusters[i][:,1], 'o', markerfacecolor=col, markersize=0.5)
                radius,xvalue,yvalue,amp,likelihood = calculateRadius(x1,y1,a1,r1,l1)
                coordsR.append(radius)
                coordsA.append(amp)
                coordsL.append(likelihood)
        else:
            radius,xvalue,yvalue,amp,likelihood = calculateRadius(x1,y1,a1,r1,l1)
            coordsX.append(xvalue)
            coordsY.append(yvalue)
            coordsR.append(radius)
            coordsA.append(amp)
            coordsL.append(likelihood)
            
    for i in range(len(coordsX)):
        circle =  plt.Circle((coordsX[i],coordsY[i]),coordsR[i],edgecolor = 'r',facecolor='none')
        fig = plt.gcf()
        fig.gca().add_artist(circle)
        #print coordsX[i],coordsY[i],coordsR[i]
    
    originalData = np.load(output_folder +"/" + prefix + "_srcs.npy")

    for i in range(len(originalData)):
        circle =  plt.Circle((originalData[i][0],originalData[i][1]),originalData[i][3],edgecolor = 'k',facecolor='none')
        fig = plt.gcf()
        fig.gca().add_artist(circle)
        #print originalData[i][0],originalData[i][1],originalData[i][3]
    plt.plot(coordsX, coordsY, 'o', markerfacecolor="r", markersize=2)
    plt.plot(originalData[:,0], originalData[:,1], 'o', markerfacecolor="k", markersize=2)
    plt.title('Estimated number of clusters: %d' % len(coordsX))
    plt.savefig(output_folder + "/clusters_active_points.png", bbox_inches="tight")
    plt.show()
    #w = make_plot("summary_active_points", X,Y,A,R,L,width,height,prefix,output_folder)
    
    temp = zeros((len(coordsX),5))
    temp[:,0] = coordsX
    temp[:,1] = coordsY
    temp[:,2] = coordsA
    temp[:,3] = coordsR
    temp[:,4] = coordsL
    #print temp
    savetxt(output_folder +"/finalData.txt", temp,fmt='%.6f')
    tp, fp, ud = post_run(output_folder,prefix)
    return tp,fp,ud
