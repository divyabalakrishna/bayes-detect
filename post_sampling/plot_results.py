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

x,y,r,a,L = loadtxt(output_folder + "/active_points.txt", unpack=True)
d = zeros((len(x),5))
d[:,0] = x
d[:,1] = y
d[:,2] = r
d[:,3] = a
d[:,4] = L
def random_color():
    return plt.cm.gist_ncar(random.random())

'''def find_locs(x, l, ax):
    
    maxL = sorted(l)[0]
    #print maxL
    for x,y,r,a,l in d:
        #if floor(maxX) == floor(x) or floor(maxX-1) == floor(x-1):
        if maxL == l:
            print x,y,r,a,l
            ax.plot(x,y,'o', color="r", markersize=10)'''
    
def plot_segments(ax, locs, vals, min_vals, max_vals, ax1):
    """
    plots each segment with a different color
    where a segment should contain one peak
    """
    intervals = compute_intervals(min_vals, max_vals)
    intervals = floor(intervals).astype("int")
    meanX = []
    print min_vals, max_vals
    for x,y in intervals:
        if(x == y):
            x = 0
        lower_mask = locs > x
        upper_mask = locs < y
        mask = logical_and(lower_mask, upper_mask)
        ax.plot(locs[mask], vals[mask], color=random_color())
        #find_locs(locs[mask],vals[mask], ax1)
        meanX.append(mean(locs[mask]))
        #color is chosen randomly, so sometimes it makes a bad selection
    x = intervals[-1][1]
    y = floor(sorted(locs)[-1]).astype("int")
    lower_mask = locs > x
    upper_mask = locs < y
    mask = logical_and(lower_mask, upper_mask)
    ax.plot(locs[mask], vals[mask], color=random_color())
    #find_locs(locs[mask],vals[mask], ax1)
    meanX.append(mean(locs[mask]))
    return meanX
#first plot of parameter vs L
print "1"
fig=plt.figure(1,figsize=(15,10), dpi=100)
ax1=fig.add_subplot(2,3,1)

ax1.scatter(x,y,s=3,marker='.')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_title('all posteriors before cut')
ax1.set_xlim(0,width)
ax1.set_ylim(0,height)

w, xmask, xm, Lmx = binned_max(x, L, 0, width, 600)
#print w
#print xmask, len(xmask), len(L)
#print sorted(xm[xmask])#,sorted(x)
#print Lmx 
print "3"
ax3=fig.add_subplot(2,3,2)

ax3.scatter(x[w],y[w],s=3,marker='.')
ax3.set_xlabel('X')
ax3.set_ylabel('Y')
ax3.set_xlim(0,width)
ax3.set_ylim(0,height)
ax3.set_title('posteriors after cut')



print "2"
ax2=fig.add_subplot(2,3,4)
ax2.plot(x[w],L[w],'k,')
ax2.set_xlabel('X')
ax2.set_ylabel('Likelihood')
#ax2.plot(xm[xmask],Lmx[xmask],'r-')
smoothed_x = smooth(Lmx[xmask])
#ax2.plot(xm[xmask], smoothed_x, 'g-')
mins = compute_mins(xm[xmask], smoothed_x)
maxes = compute_maxes(xm[xmask], smoothed_x)
"""
#plots vertical lines
[ax2.axvline(x = val, c="b") for val in mins]
[ax2.axvline(x = val, c="r") for val in maxes]
"""
mean_x=plot_segments(ax2, xm[xmask], Lmx[xmask], mins, maxes, ax3)
ax2.set_title('X vs Likelhood after cut')

w, ymask, ym, Lmy = binned_max(y, L, 0, height, 600)

print "4"
ax4=fig.add_subplot(2,3,5)
ax4.plot(y[w],L[w],'k,')
ax4.set_xlim(0, width)
ax4.set_xlabel('Y')
ax4.set_ylabel('Likelihood')
#ax4.plot(ym[ymask],Lmy[ymask],'r-')
smoothed_y = smooth(Lmy[ymask])
#ax4.plot(ym[ymask], smoothed_y, 'g-')

mins = compute_mins(ym[ymask], smoothed_y)
maxes = compute_maxes(ym[ymask], smoothed_y)
"""
[ax4.axvline(x = val, c="b") for val in mins]
[ax4.axvline(x = val, c="r") for val in maxes]
"""
mean_y=plot_segments(ax4, ym[ymask], Lmy[ymask], mins, maxes, ax3)

ax4.set_title('Y vs Likelhood after cut')

print "5"
ax5 = fig.add_subplot(2,3,3)
data = load(output_folder + "/0_clean.npy")
ax5.imshow(flipud(data),extent=[0,width,0,height])
ax5.set_title('Original image ')

for i in range(len(mean_x)):
    Y = []
    X = []
    j = 0
    while(j<height):
        Y.append(j)
        j+=1
        X.append(mean_x[i])
    ax5.plot(X,Y)
for i in range(len(mean_y)):
    Y = []
    X = []
    j = 0
    while(j<height):
        Y.append(mean_y[i])
        X.append(j)
        j+=1
    ax5.plot(X,Y)

print "save"
#plt.savefig('plots/summary.png',bbox_inches='tight')
plt.savefig(output_folder + "/plots/summary_active.png", bbox_inches="tight")


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
plt.savefig(output_folder + "/plots/3d_posterior.png", bbox_inches="tight")

print "display"
#plt.show()

