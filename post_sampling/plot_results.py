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

x,y,r,a,L = loadtxt(output_folder + "/" + prefix + "_out_points_som.txt", unpack=True)

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

#first plot of parameter vs L
print "1"
fig=plt.figure(figsize=(14,8))
ax1=fig.add_subplot(2,3,2)

ax1.scatter(x,y,s=3,marker='.')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_title('all posteriors before cut')
ax1.set_xlim(0,width)
ax1.set_ylim(0,height)

w, xmask, xm, Lmx = binned_max(x, L, 0, width, 350)

print "2"
ax2=fig.add_subplot(2,3,1)
ax2.plot(x[w],L[w],'k,')
ax2.set_xlabel('X')
ax2.set_ylabel('Likelihood')
ax2.plot(xm[xmask],Lmx[xmask],'r-')
smoothed_x = smooth(Lmx[xmask])
ax2.plot(xm[xmask], smoothed_x, 'g-')
mins = compute_mins(xm[xmask], smoothed_x)
maxes = compute_maxes(xm[xmask], smoothed_x)
"""
#plots vertical lines
[ax2.axvline(x = val, c="b") for val in mins]
[ax2.axvline(x = val, c="r") for val in maxes]
"""
plot_segments(ax2, xm[xmask], smoothed_x, mins, maxes)
ax2.set_title('X vs Likelhood after cut')


print "3"
ax3=fig.add_subplot(2,3,3)

ax3.scatter(x[w],y[w],s=3,marker='.')
ax3.set_xlabel('X')
ax3.set_ylabel('Y')
ax3.set_xlim(0,width)
ax3.set_ylim(0,height)
ax3.set_title('posteriors after cut')

w, ymask, ym, Lmy = binned_max(y, L, 0, height, 350)

print "4"
ax4=fig.add_subplot(2,3,4)
ax4.plot(y[w],L[w],'k,')
ax4.set_xlim(0, width)
ax4.set_xlabel('Y')
ax4.set_ylabel('Likelihood')
ax4.plot(ym[ymask],Lmy[ymask],'r-')
smoothed_y = smooth(Lmy[ymask])
ax4.plot(ym[ymask], smoothed_y, 'g-')

mins = compute_mins(ym[ymask], smoothed_y)
maxes = compute_maxes(ym[ymask], smoothed_y)
"""
[ax4.axvline(x = val, c="b") for val in mins]
[ax4.axvline(x = val, c="r") for val in maxes]
"""
plot_segments(ax4, ym[ymask], smoothed_y, mins, maxes)

ax4.set_title('Y vs Likelhood after cut')

w, rmask, rm, Lmr = binned_max(r, L, rad_min, rad_max, 350)
    
print "5"
ax5=fig.add_subplot(2,3,5)
ax5.plot(r[w],L[w],'k,')
ax5.set_xlim(rad_min, rad_max)
ax5.set_xlabel('R')
ax5.set_ylabel('Likelihood')
ax5.plot(rm[rmask],Lmr[rmask],'r-')
smooth_r = smooth(Lmr[rmask])
ax5.plot(rm[rmask], smooth_r, 'g-')
ax5.set_title('R vs Likelhood after cut')


w, amask, am, Lma = binned_max(a, L, amp_min, amp_max, 350)

print "6"
ax6=fig.add_subplot(2,3,6)
ax6.plot(a[w],L[w],'k,')
ax6.set_xlim(amp_min, amp_max)
ax6.set_xlabel('A')
ax6.set_ylabel('Likelihood')
ax6.plot(am[amask],Lma[amask],'r-')
smooth_a = smooth(Lma[amask])
ax6.plot(am[amask], smooth_a, 'g-')
ax6.set_title('A vs Likelhood after cut')

print "save"
#plt.savefig('plots/summary.png',bbox_inches='tight')
plt.savefig(output_folder + "/plots/summary.png", bbox_inches="tight")


#second plot of 3d parameters (x,y) vs L
fig= plt.figure()

proj = fig.add_subplot(111, projection='3d')
proj.scatter(x[w],y[w],L[w],s=3,marker='.')
proj.set_xlim(0,width)
proj.set_ylim(0,height)
proj.set_xlabel('X')
proj.set_ylabel('Y')
proj.set_zlabel('Likelihood')
#proj.set_title('Posteriors in 3D after cut')
plt.savefig(output_folder + "/plots/3dPosterior.png", bbox_inches="tight")

print "display"
#plt.show()


#Mean-Shift cluster
X=zeros((len(w),2))
X[:,0]=x[w]
X[:,1]=y[w]

# The following bandwidth can be automatically detected using
bandwidth = estimate_bandwidth(X, quantile=0.1, n_samples=300)

ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
ms.fit(X)
labels = ms.labels_
cluster_centers = ms.cluster_centers_

labels_unique = np.unique(labels)
n_clusters_ = len(labels_unique)

print("number of estimated clusters : %d" % n_clusters_)
plt.title('Estimated number of clusters: %d' % n_clusters_)

plt.figure(1)
plt.clf()

colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
for k, col in zip(range(n_clusters_), colors):
    my_members = labels == k
    cluster_center = cluster_centers[k]
    plt.plot(X[my_members, 0], X[my_members, 1], col + '.')
    plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor='#ffffff',
             markeredgecolor='k', markersize=15)
plt.title('Estimated number of clusters: %d' % n_clusters_)

plt.savefig(output_folder + "/plots/clusters.png", bbox_inches="tight")



