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

prefix = 107
configfile = "../files"+"/"+str(prefix)+"/"+'config'+'_'+str(prefix)+'.ini'

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

x,y,r,a,L = loadtxt("../"+output_folder + "/" + prefix + "_out_points_som.txt", unpack=True)
Lsmooth = smooth(L)