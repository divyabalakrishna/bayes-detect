import numpy as np
from scipy.signal import argrelmin,argrelmax

def smooth(values, window_len = 7, window="flat"):
    #window_len must be odd
    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'" 

    s = np.r_[values[window_len-1:0:-1],values,values[-1:-window_len:-1]]
    if window == "flat":
        w=np.ones(window_len,'d')
    else:
        w=np.eval(window+'(window_len)')
    result=np.convolve(w/w.sum(),s,mode='valid')
    return result[:values.shape[0]]

def binned_max(xvalues, yvalues, start, stop, num_points):
    points=np.linspace(start,stop,num_points+1)
    bins=0.5*(points[1:]+points[:-1])
    Lval=np.zeros(len(bins))
    bin_width=points[1]-points[0] #compute size of each bin
    idx=np.floor(xvalues/bin_width) #reindex into bins
    idx=idx.astype('int')
    for i in xrange(len(bins)):
        wi=np.where(idx == i)[0] #getting points inside the bin
        if np.shape(wi)[0] == 0: continue
        #if theres nothing there just continue
        else:
            Lval[i]=max(yvalues[wi]) #bin's value is the biggest value in the bin

    mask= (Lval != 0.)
    w=np.where(yvalues > min(Lval))[0]
    return (w, mask, bins, Lval)

def compute_mins(xlocs, yvals, window_size = 10):
    yval_locs = argrelmin(yvals, order = window_size)[0]
    if len(yval_locs) == 0:
        return [];
    minPoints = xlocs[yval_locs]
    list1 = []
    for i in minPoints:
        list1.append(i)
    return list1

def compute_maxes(xlocs, yvals, window_size = 10):
    yval_locs = argrelmax(yvals, order = window_size)[0]
    if len(yval_locs) == 0:
        return [];
    maxPoints = xlocs[yval_locs]
    list1 = []
    for i in maxPoints:
        list1.append(i)
    return list1 

def compute_intervals(mins, maxes):
    #first val indicates its dim
    minvals = zip([0 for x in xrange(len(mins))], mins)
    maxvals = zip([1 for x in xrange(len(maxes))], maxes)
    
    #merge them
    all_vals = minvals + maxvals
    all_vals.sort(key = lambda x: x[-1]) 
    
    points = []
    index = 0
    start = None

    #get the start
    for x in xrange(len(all_vals)):
        if all_vals[x][0] == 0:
            start = all_vals[x][1]
            break

    seen_max = False

    #each min, max, ..., min is one interval
    while index < len(all_vals):
        if seen_max and all_vals[index][0] == 0:
            #we found the next min
            points.append([start, all_vals[index][1]])
            start = all_vals[index][1]
            seen_max = False
        elif not seen_max and all_vals[index][0] == 1:
            #now we have seen a max
            seen_max = True
        index += 1

    return np.array(points)
