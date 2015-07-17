from numpy import *
import scipy
import os,sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import DBSCAN
from sklearn import metrics

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MeanShift, estimate_bandwidth
from ConfigParser import SafeConfigParser
from itertools import cycle
from scipy.signal import argrelextrema

import seaborn as sns #makes the plots look pretty

from common import *

try:
    os.mkdir('plots')
except:
    pass

parser = SafeConfigParser()
parser.read("../config.ini")

width = int(parser.get("Sampling", "width"))
height = int(parser.get("Sampling", "height"))

amp_min = float(parser.get("Sampling", "amp_min"))
amp_max = float(parser.get("Sampling", "amp_max"))

rad_min = float(parser.get("Sampling", "rad_min"))
rad_max = float(parser.get("Sampling", "rad_max"))

prefix = parser.get("Misc", "prefix")
location = parser.get("Misc", "location")
output_folder = location + "/" + prefix 

x,y,r,a,l = loadtxt(output_folder + "/active_points.txt", unpack=True)
x_p,y_p,r_p,a_p,l_p = loadtxt(output_folder + "/0_out_points_som.txt", unpack=True)

def dbscan(XX,name,length,breath,min_samples):
    #DBSCAN

    N = len(XX[:,0])
    eps = 2*(sqrt((length*breath)/N))
    
    db = DBSCAN(eps=eps).fit(XX)

    core_samples_mask = zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_


    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)


    print n_clusters_ , 'Clusters'

    unique_labels = set(labels)
    colors = plt.cm.jet(linspace(0, 1, len(unique_labels)))

    clusters = [XX[labels == i] for i in xrange(n_clusters_)]
    plt.figure()

    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = 'k'
            continue

        class_member_mask = (labels == k)
        xy = XX[class_member_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
             markeredgecolor='k', markersize=5)
    

    plt.title('Estimated number of clusters: %d' % n_clusters_)
    plt.savefig(output_folder + "/plots/"+name+".png", bbox_inches="tight")
    
    return clusters

#clusters of active points
XX=zeros((len(x),2))
XX[:,0]=x
XX[:,1]=y
dbscan(XX,"clusters_active_points",width,height,5)

#to find the minimum Likelihood in active points
minL = l[0]
for i in l:
    if(i < minL):
        minL = i

X = []
Y = []
L = []
R = []
A = []

for i in range(len(l_p)):
    if(l_p[i] > minL):
        X.append(x_p[i])
        Y.append(y_p[i])
        R.append(r_p[i])
        A.append(a_p[i])
        L.append(l_p[i])

XX = zeros((len(X),2))
XX[:,0] = X
XX[:,1] = Y
clusters = dbscan(XX, "clusters_posterior",width,height,5)

for i in range(len(clusters)):
    minX = clusters[i][0][0]
    maxX = clusters[i][0][0]
    minY = clusters[i][0][1]
    maxY = clusters[i][0][1]
    for j in range(len(clusters[i])):
        if(minX > clusters[i][j][0]):
            minX = clusters[i][j][0]
        if(maxX < clusters[i][j][0]):
            maxX = clusters[i][j][0]
        if(minY > clusters[i][j][1]):
            minY = clusters[i][j][1]
        if(maxY < clusters[i][j][1]):
            maxY = clusters[i][j][1]
    dbscan(clusters[i],"clusters"+str(i),maxX-minX,maxY-minY,5)
