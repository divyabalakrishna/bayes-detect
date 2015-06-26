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

def random_color():
    return plt.cm.gist_ncar(random.random())

def plot_segments(ax, locs, vals, min_vals, max_vals):
    """
    plots each segment with a different color
    where a segment should contain one peak
    """
    intervals = compute_intervals(min_vals, max_vals)
    intervals = floor(intervals).astype("int")
    for x,y in intervals:
        lower_mask = locs > x
        upper_mask = locs < y
        mask = logical_and(lower_mask, upper_mask)
        ax.plot(locs[mask], vals[mask], color=random_color())
        #color is chosen randomly, so sometimes it makes a bad selection

def get_likelihood(X,Y,name):
    L = []
    R = []
    A = []
    for i in range(len(X)):
        for j in range(len(x)):
            if(X[i] == x[j] and Y[i] == y[j]):
                #print i,j
                L.append(l[j])
                R.append(r[j])
                A.append(a[j])
    cluster = zeros((len(X),5))
    cluster[:,0] = X
    cluster[:,1] = Y
    cluster[:,2] = R
    cluster[:,3] = A
    cluster[:,4] = L
    
    x_p = []
    y_p = []
    r_p = []
    a_p = []
    l_p = []

    minX = X[0]
    maxX = X[0]
    for i in range(len(X)):
        if(minX > X[i]): minX = X[i]
        if(maxX < X[i]): maxX = X[i]
    
    minY = Y[0]
    maxY = Y[0]
    for i in range(len(Y)):
        if(minY > Y[i]): minY = Y[i]
        if(maxY < Y[i]): maxY = Y[i]
    print minX,maxX,minY,maxY
    
    for i in range(len(x)):
        if(x[i] >= minX and x[i] <=maxX and y[i] >=minY and y[i] <=maxY):
            x_p.append(x[i])
            y_p.append(y[i])
            r_p.append(r[i])
            a_p.append(a[i])
            l_p.append(l[i])
    cluster_p = zeros((len(x_p),5))
    cluster_p[:,0] = x_p
    cluster_p[:,1] = y_p
    cluster_p[:,2] = r_p
    cluster_p[:,3] = a_p
    cluster_p[:,4] = l_p
    savetxt(output_folder + "/cluster"+str(name)+".txt", cluster_p,fmt='%.6f')
    x1,y1,r1,a1,l1 = loadtxt(output_folder + "/cluster"+str(name)+".txt", unpack=True)
    #print x1
    #print l_p
    make_plot("cluster"+str(name), x1,y1,r1,a1,l1)
    return cluster

def make_plot(filename, x, y, r, a, l):
    #first plot of parameter vs L
    print "1"
    fig=plt.figure(figsize=(10,8))
    ax1=fig.add_subplot(2,2,1)

    ax1.scatter(x,y,s=3,marker='.')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_title('all posteriors before cut')
    ax1.set_xlim(0,width)
    ax1.set_ylim(0,height)

    w, xmask, xm, Lmx = binned_max(x, l, 0, width, 350)

    print "2"
    ax2=fig.add_subplot(2,2,3)
    ax2.plot(x[w],l[w],'k,')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Likelihood')
    ax2.plot(xm[xmask],Lmx[xmask],'r-')
    smoothed_x = smooth(Lmx[xmask])
    ax2.plot(xm[xmask], smoothed_x, 'g-')
    #mins = compute_mins(xm[xmask], smoothed_x)
    #maxes = compute_maxes(xm[xmask], smoothed_x)
   
    #plot_segments(ax2, xm[xmask], smoothed_x, mins, maxes)
    ax2.set_title('X vs Likelhood after cut')

    print "3"
    ax3=fig.add_subplot(2,2,2)

    ax3.scatter(x[w],y[w],s=3,marker='.')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_xlim(0,width)
    ax3.set_ylim(0,height)
    ax3.set_title('posteriors after cut')

    w, ymask, ym, Lmy = binned_max(y, l, 0, height, 350)

    print "4"
    ax4=fig.add_subplot(2,2,4)
    ax4.plot(y[w],l[w],'k,')
    ax4.set_xlim(0, width)
    ax4.set_xlabel('Y')
    ax4.set_ylabel('Likelihood')
    ax4.plot(ym[ymask],Lmy[ymask],'r-')
    smoothed_y = smooth(Lmy[ymask])
    ax4.plot(ym[ymask], smoothed_y, 'g-')

    #mins = compute_mins(ym[ymask], smoothed_y)
    #maxes = compute_maxes(ym[ymask], smoothed_y)
    
    #plot_segments(ax4, ym[ymask], smoothed_y, mins, maxes)

    ax4.set_title('Y vs Likelhood after cut')

    print "save"

    plt.savefig(output_folder + "/plots/"+filename+".png", bbox_inches="tight")
    return w

w = make_plot("summary_active_points", x, y, r, a, l)
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
plt.savefig(output_folder + "/plots/3dPosterior_active_points.png", bbox_inches="tight")

print "display"
#plt.show()

#DBSCAN

XX=zeros((len(w),2))
XX[:,0]=x[w]
XX[:,1]=y[w]
#XX[:,2]=r[w]
#XX[:,3]=a[w]
#XX[:,2]=l[w]

#XX = StandardScaler().fit_transform(XX)

db = DBSCAN(eps=5,min_samples=5).fit(XX)

core_samples_mask = zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_


n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)


print n_clusters_ , 'Clusters'

unique_labels = set(labels)
colors = plt.cm.jet(linspace(0, 1, len(unique_labels)))

plt.figure()

#store cluster points 
clusters = [XX[labels == i] for i in xrange(n_clusters_)]

for i in range(len(clusters)):
    clusters[i] = get_likelihood(clusters[i][:,0], clusters[i][:,1], i)
    #make_plot("cluster"+str(i), clusters[i][:,0],clusters[i][:,1],clusters[i][:,2],clusters[i][:,3],clusters[i][:,4])

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
plt.savefig(output_folder + "/plots/clusters_active_points.png", bbox_inches="tight")

