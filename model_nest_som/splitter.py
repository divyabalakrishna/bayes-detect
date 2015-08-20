import numpy as np
import scipy
import common

#we only look at the x and y dimensions
#which happen to be 0, and 1 in the array
def next_dim(dim):
    if dim == 0:
        return 1
    return 0

def get_peaks(all_vals, initial_bounds):
    bin_amt = 350 #we hardcoded the bin amt, but this can be changed
    queue = []
    results = [] #going to store results as (depth, [xmin, xmax, ymin, ymax])

    L = all_vals[4]
    
    queue.append((0, 0, initial_bounds))#depth, dim, [[xlower, xupper],[ylower, yupper]]
    while queue != []:
        depth, dim, bounds = queue.pop()
        start, stop = bounds[dim]
        other_start, other_stop = bounds[next_dim(dim)]
        
        if start == stop or other_start == other_stop:
            #if one of the bounds is actually one point, then we treat it as a detected source
            #todo: look into if this is the cause of various false positives
            results.append((depth, bounds.flatten()))
            continue
        
        dimvals = all_vals[dim]
        range_mask = np.where((dimvals >= start) & (dimvals <= stop))[0]
        dimvals = dimvals[range_mask]
        lvals = all_vals[4, range_mask]
        
        
        _, main_mask, main_binned, main_binned_L = common.binned_max(dimvals, lvals, start, stop, bin_amt) 
        
        if main_binned_L[main_mask].shape[0] == 0:
            #there nothing here
            continue
        main_smoothed = common.smooth(main_binned_L[main_mask])
        
        #check if there is a peak
        median = np.median(main_smoothed)
        peak = np.max(main_smoothed)
        if peak < 0.999 * median:
            continue
        else:
            results.append((depth, bounds.flatten()))
        
        window_size = 2

        main_mins = common.compute_mins(main_binned[main_mask], main_smoothed, window_size=window_size)
        main_maxes = common.compute_maxes(main_binned[main_mask], main_smoothed, window_size=window_size)
        main_intervals = common.compute_intervals(main_mins, main_maxes)
        main_intervals = np.floor(main_intervals).astype("int")
        
        if main_intervals.shape[0] == 0: #no intervals to look at
            continue
        
        for nstart, nstop in main_intervals:
            if nstart == nstop:
                continue
        
            other_col = all_vals[next_dim(dim), :]
            my_col = all_vals[dim, :]
            my_mask = np.where((my_col >= nstart) & (my_col <= nstop))[0]
        
            _, my_mask, my_binned, my_binned_L = common.binned_max(other_col[my_mask], L[my_mask], other_start, other_stop, 50) 
        
        
            if my_binned_L[my_mask].shape[0] == 0:
                continue
        
            my_smoothed = common.smooth(my_binned_L[my_mask])

            my_mins = common.compute_mins(my_binned[my_mask], my_smoothed, window_size=5)
            my_maxes = common.compute_maxes(my_binned[my_mask], my_smoothed, window_size=5)
            my_intervals = common.compute_intervals(my_mins, my_maxes)
            my_intervals = np.floor(my_intervals).astype("int")
            
            for my_start, my_stop in my_intervals:
                b = np.zeros((2,2))
                b[dim] = nstart, nstop
                b[next_dim(dim)] = my_start, my_stop
                queue.append((depth+1, next_dim(dim), b))
                #split on the other dimension but increment the depth
                
    return results

def get_sources(s, all_vals):
    
    sources = np.zeros((len(s), 6)) #x,y,a,r,depth,l
    x = all_vals[0]
    y = all_vals[1]
    r = all_vals[2]
    a = all_vals[3]
    L = all_vals[4]
    
    for i in xrange(len(s)):
        depth, (xlower, xupper, ylower, yupper) = s[i]
        mask = np.where((x >= xlower) & (x <= xupper) & (y >= ylower) & (y <= yupper))[0]
        if L[mask].shape[0] == 0:
            continue
        #we take the x,y,a,r,L values from the point in the range with the highest likelihood
        maxindex = np.argmax(L[mask])
        sources[i, 0] = x[mask][maxindex]
        sources[i, 1] = y[mask][maxindex]
        sources[i, 2] = r[mask][maxindex]
        sources[i, 3] = a[mask][maxindex]
        sources[i, 4] = depth
        sources[i, 5] = L[mask][maxindex]
    sources = sources[~np.isnan(sources).any(axis=1)]
    #filter out NaNs
    return sources

#make sources is taken from image_gen
def make_source(src_array, height, width):

    x = np.arange(0, width)
    y = np.arange(0, height)
    xx, yy = np.meshgrid(x, y, sparse=True)
    z = np.zeros((height,width),float)
    for i in src_array:
        z+= i[2]*np.exp(-1*((xx-i[0])**2+(yy-i[1])**2)/(2*(i[3]**2)))
    return z


