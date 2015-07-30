"""
.. module:: model_nest_som
.. moduleauthor:: Matias Carrasco Kind

"""
__author__ = 'Matias Carrasco Kind'

"""
Many things to do, and comment and fix
This is preliminary
"""
from numpy import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#import pyfits as pf
import sys,os
import scipy
import timeit
from scipy import stats
import SOMZ as som
import pickle
import matplotlib.cm as cm
from scipy.spatial import Voronoi,voronoi_plot_2d
import copy
from scipy.spatial import ConvexHull
from matplotlib.patches import Circle, Wedge, Polygon
from matplotlib.collections import PatchCollection
import matplotlib.image as mpimg
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn import metrics
from ConfigParser import SafeConfigParser

from splitter import get_peaks, get_sources, make_source

from scipy.spatial import distance

highestClusterCount = {}
highestClusterCount["count"] = 0
highestClusterCount["iteration"] = 0
clusters = []

"""
Given our array of active points, we now try to detect sources in it.
arr = active points
iteration = the iteration
detected = the numpy array of detected points
detected_count = how many times this item has been detected
n_dist = the maximum distance two points can be from each other before they are considered the same

The idea is we can keep track of detected items, and then look at the top K
deteted items. If they appear many times, it is much more likely that 
they are real sources and not noise points.
"""
def detect_sources(arr, iteration, detected, detected_count, n_dist):
    arr = arr.T
    
    #just some setup for running the splitter
    initial_bounds = array([[0, width], [0, height]])
    k = get_peaks(arr, initial_bounds)
    #run the splitter
    sources = get_sources(k, arr)
    sources = sources[~all(sources == 0, axis=1)] #remove 0 rows
    #the splitter sometimes returns rows of 0s which are meaningless, so we remove them
    
    #sources are the items we have found in this iteration
    #detected are the items we detected earlier
    xy_index = array([0,1]) #we want to look at the x,y positions only which happen to be col 0 and 1
    distances = distance.cdist(sources[:,xy_index], detected[:, xy_index]) #cdist is faster than looping over each item
    """
    distances is a array of euclidian distances in the format
    [[s1 to d1, s1 to d2, ..., s1 to dn], [s2 to s1, s2 to d2, ..., s2 to dn], [sk to d1, ..., sk to dn]]
    where we have k rows/sources in sources and n rows/sources in detected
    """
    
    for s_idx in xrange(distances.shape[0]):
        #iterate over each "new" source
        points_within_range = (distances[s_idx] < n_dist) #checks to see if any of the distances are below the n_dist
        within_range = any(points_within_range) #if any of them are it will evaluate to true

        if within_range:
            #find which one it is closest to and increment that one's counter
            relevant_distances = sort(distances[s_idx, points_within_range]) #get the distances within our range and then sort it
            index_of_closest_value = where(distances[s_idx] == relevant_distances[0])[0] #index (col) of the closest item.
            #where returns a tuple so we need to extract our value
            detected_count[index_of_closest_value[0]] += 1 #add to its count of detections

        else:
            detected = vstack((detected, sources[s_idx])) #put the new item on the bottom of the array
            detected_count.append(1) #add a counter for it
    
    """
    #for each detected source either increment its detections or store it with 1 detection
    for s_idx in xrange(sources.shape[0]):
        check_temp = (detected == sources[s_idx]).all(axis=1)
        #check the detected source matches a row in our already known sources
        if any(check_temp):
            #we already have this value
            pos = where(check_temp == True)[0] #get the position of the matching row
            detected_count[pos] += 1 #add to its count of detections

        else:
            #put the new source on the bottom of the detected list
            #add its detected count with the value 1
            detected = vstack((detected, sources[s_idx]))
            detected_count.append(1)
    """

    
    dc = array(detected_count)
    d = c_[detected, dc] #concatenate them with dc as the last column
    d = d[d[:,-1].argsort()] #sort it by detected count (the last column)
    myindex = array([0,1,-1]) #array slicing thing, 0 = x, 1 = y, -1 = the last col/the counts
    print d[:, myindex] #we only want to print x,y,#hits

    return sources[:, 0:4], detected, detected_count

"""
Given our list of sources and the iteration #
We make the image for it and store its iteration# in its filename
"""
def look_at_results(sources, iteration):
    img = make_source(sources, height, width)
    
    fig = plt.figure()    
    ax1 = fig.add_subplot(111)
    #we show the image, but we need to flip it because of a difference in how the image is shown
    #and how the image is stored
    ax1.imshow(flipud(img),extent=[0,width,0,height], cmap="jet")
    ax1.set_title("detected")
    ax1.set_xlim(0, width)
    ax1.set_ylim(0, height)
    ax1.grid(False)

    fname="%05d" % iteration  #getting the 5 digit represetation of the iteraiton number 
    fig.savefig(output_folder + '/plots/detected/' + fname + '.png',bbox_inches='tight') #store it

def voronoi_plot_2d_local(vor, ax=None):
    ver_all=vor.vertices
    ncurr=len(vor.vertices)
    ntot=len(vor.vertices)
    nfix=len(vor.vertices)
    for simplex in vor.ridge_vertices:
        simplex = asarray(simplex)
        if all(simplex >= 0):
            ax.plot(vor.vertices[simplex,0], vor.vertices[simplex,1], 'k-', lw=2)
    ptp_bound = vor.points.ptp(axis=0)
    center = vor.points.mean(axis=0)
    for pointidx, simplex in zip(vor.ridge_points, vor.ridge_vertices):
        simplex = asarray(simplex)
        if any(simplex < 0):
            i = simplex[simplex >= 0][0]  # finite end Voronoi vertex
            t = vor.points[pointidx[1]] - vor.points[pointidx[0]]  # tangent
            t /= linalg.norm(t)
            n = array([-t[1], t[0]])  # normal
            midpoint = vor.points[pointidx].mean(axis=0)
            direction = sign(dot(midpoint - center, n)) * n
            far_point = vor.vertices[i] + direction * ptp_bound.max()
            ax.plot([vor.vertices[i,0], far_point[0]],
                    [vor.vertices[i,1], far_point[1]], 'k-',lw=2)
            ver_all=concatenate((ver_all,[[far_point[0],far_point[1]]]))
    for s in vor.ridge_vertices:
        if -1 in s:
            s[s.index(-1)]=ncurr
            ncurr+=1
    return ver_all

"""
Given iteration number and cluster count at an iteration
a 2d plot is drawn
"""
def make_clusterCountPlot(clusterCount):
    fig= plt.figure()
    proj = fig.add_subplot(111)
    proj.plot(clusterCount[:,0],clusterCount[:,1],'o',color='#000000',markersize=10)
    proj.set_xlim(0,width)
    proj.set_ylim(0,height)
    proj.set_xlabel('Iteration number')
    proj.set_ylabel('Cluster Count')
    
    fig.savefig(output_folder + '/plots/clusterCount.png',bbox_inches='tight')


"""
Given active points (AC) and the iteration number (name)
3d plots are drawn with parameters (x,y) vs L
"""
def make_3dplot(AC,name):
    #plot of 3d parameters (x,y) vs L
    fig= plt.figure()

    proj = fig.add_subplot(111, projection='3d')
    proj.scatter(AC[:,0],AC[:,1],AC[:,4],s=3,marker='.')
    proj.set_xlim(0,width)
    proj.set_ylim(0,height)
    proj.set_xlabel('X')
    proj.set_ylabel('Y')
    proj.set_zlabel('Likelihood')
    
    
    fname="%05d" % name

    fig.savefig(output_folder + '/plots/3dplot/'+fname+'.png',bbox_inches='tight')

"""
Given our sampled points (points) and active points (AC) and the iteration number (name)
We make various plots (more detail in their titles)
"""
def make_plot(points,AC,name):
    global clusters
    fig=plt.figure(1,figsize=(15,10), dpi=100)
    ax1=fig.add_subplot(2,3,1)  
    ax1.plot(points[:name,0],points[:name,1],'k.')
    ax1.set_xlim(0,width)
    ax1.set_ylim(0,height)
    ax1.set_title('Posterior points')
    ax1.set_yticks([])
    ax1.set_xticks([])
    #just plotting the active points


    
    '''ax3=fig.add_subplot(2,3,2)
    xt=points[:name,0]
    yt=points[:name,1]
    hh,locx,locy=scipy.histogram2d(xt,yt,bins=[linspace(0,width,width+1),linspace(0,height,height+1)])
    #x,y histogram
    #more common points will increase the histogram value and be more intense on the image
    ax3.imshow(flipud(hh.T),extent=[0,width,0,height],aspect='normal')
    ax3.set_title('Pseudo image from posterior')
    '''
    #DBSCAN

    XX=zeros((len(AC[:,0]),2))
    XX[:,0]=AC[:,0]
    XX[:,1]=AC[:,1]
    XX = StandardScaler().fit_transform(XX)
 
    N = len(XX[:,0])
    eps = 2*(200/sqrt(N))
    db = DBSCAN().fit(XX)

    core_samples_mask = zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_


    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    unique_labels = set(labels)
    colors = plt.cm.jet(linspace(0, 1, len(unique_labels)))

    ax3=fig.add_subplot(2,3,2)
    
    storeClusters = False
    if(highestClusterCount["count"] <= n_clusters):
        savetxt(output_folder + "/active_points.txt", AC,fmt='%.6f')
        highestClusterCount["count"] = n_clusters
        highestClusterCount["iteration"] = name
        storeClusters = True
        clusters = []

    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = 'k'
            continue

        class_member_mask = (labels == k)
        xy = XX[class_member_mask]
        ax3.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
             markeredgecolor='k', markersize=5)
        
        if(storeClusters and k != -1):
            clusters.append(xy)
    
    ax3.set_title('Estimated number of clusters: %d' % n_clusters)


    ax2=fig.add_subplot(2,3,3)
    ax2.plot(AC[:,0],AC[:,1],'k.')
    ax2.set_xlim(0,width)
    ax2.set_ylim(0,height)
    ax2.set_yticks([])
    ax2.set_xticks([])
    ax2.set_title('Active points')
    #just plotting the active points



    ax4=fig.add_subplot(2,3,4)
    ax4.imshow(flipud(data),extent=[0,width,0,height])
    ax4.set_title('Original image with noise')
    #show input noised image


    ax5=fig.add_subplot(2,3,5)
    ax5.imshow(flipud(data_or),extent=[0,width,0,height])
    ax5.set_title('Original image ')
    #show the original image

    fname="%05d" % name


    ax6=fig.add_subplot(2,3,6)
    img=mpimg.imread(output_folder + '/plots/somplot/som_'+fname+'.png')
    #earlier we created the som image. now we read it back in
    ax6.imshow(img,extent=[0,width,0,height],aspect='normal')
    #and display it as one of the panels in the subplot
    ax6.set_title('SOM map ')

    fig.savefig(output_folder + '/plots/6plot/all6_'+fname+'.png',bbox_inches='tight')
    fig.clear()

    """
    ax1 = orig
    ax2 = orig w/ noise
    ax3 = posterior
    ax4 = active
    """
    
    '''
    fig=plt.figure(2,figsize=(50,50), dpi=100)
    
    ax1 = plt.subplot2grid((2,2), (0,0))
    ax1.imshow(flipud(data_or),extent=[0,width,0,height])
    ax1.set_title('Original image')
    ax1.set_yticks([])
    ax1.set_xticks([])

    ax2 = plt.subplot2grid((2,2), (0,1))
    ax2.imshow(flipud(data),extent=[0,width,0,height])
    ax2.set_title('Original image with noise')
    ax2.set_yticks([])
    ax2.set_xticks([])

    ax3 = plt.subplot2grid((2,2), (1,0))
    ax3.plot(points[:name,0],points[:name,1],'k.')
    ax3.set_xlim(0,width)
    ax3.set_ylim(0,height)
    ax3.set_title('Posterior points')
    ax3.set_yticks([])
    ax3.set_xticks([])

    ax4 = plt.subplot2grid((2,2), (1,1))
    ax4.plot(AC[:,0],AC[:,1],'k.')
    ax4.set_xlim(0,width)
    ax4.set_ylim(0,height)
    ax4.set_yticks([])
    ax4.set_xticks([])
    ax4.set_title('Active points')

    fig.savefig(output_folder + '/plots/4plot/4plot'+fname+'.png',bbox_inches='tight')
    fig.clear()
    #all these just show various panels which were already shown earlier
   '''
    return n_clusters
#make source. this is the same as image_gen's make source
def make_source(src_array,height,width):
    x = arange(0, width)
    y = arange(0, height)
    xx, yy = meshgrid(x, y, sparse=True)
    z = zeros((height,width),float)
    for i in src_array:
        z+= i[2]*exp(-1*((xx-i[0])**2+(yy-i[1])**2)/(2*(i[3]**2)))
    return z

#add gaussian noise. this is the same as image_gen's adding of noise
def add_gaussian_noise(mean, sd, data):
    height = len(data)
    width = len(data[0])
    my_noise=stats.distributions.norm.rvs(mean,sd,size=(height, width))
    noised = data + my_noise
    return noised

def lnlike(a,D,nlog=0):
    X=a[0]
    Y=a[1]
    A=a[2]
    R=a[3]
    noise=abs(noise_lvl)

    if X < 0 or X > width: return [-inf, nlog]
    if Y < 0 or Y > width: return [-inf, nlog]
    if A < amp_min or A > amp_max: return [-inf, nlog]
    if R < rad_min or R > rad_max: return [-inf, nlog]
    #if any of them are out of the range, we return -inf (log(0)) and the same number of log evaluations

    S=A*exp(-(((xx-X)**2+(yy-Y)**2))/(2.*R*R))
    DD=data-S
    DDf=DD.flatten()
    Like=-0.5*linalg.norm(DDf)**2*(1./noise) - (width * height)/2 * log(2*pi) + 4 * log(noise)
    #evaluation of the log likelihood function. look in literature for the eq
    nlog+=1 #increment the log evaluation counter
    return [Like,nlog]


def sample():
    xt=random.rand()*(width - 1.)
    yt=random.rand()*(height - 1.)
    at=random.rand()*(amp_max - amp_min) + amp_min
    rt=random.rand()*(rad_max - rad_min) + rad_min
    #random point in the specified ranges
    return array([xt,yt,at,rt])


def sample_som(jj,active,neval,LLog_min,nt=5,nit=100,create='no',sample='yes',inM=''):
    if create=='yes':
        DD=array([active[:,0],active[:,1],active[:,2],active[:,3]]).T
        lmin=min(active[:,4])
        L=active[:,4]-lmin
        lmax=max(L)
        L=L/max(L)
        M=som.SelfMap(DD,L,Ntop=nt,iterations=nit,periodic='no')
        M.create_mapF()
        M.evaluate_map()
        M.logmin=lmin
        M.logmax=lmax
        ML=zeros(nt*nt)
        for i in xrange(nt*nt):
            if M.yvals.has_key(i):
                ML[i]=mean(M.yvals[i])
        M.ML=ML
        ss=argsort(ML)
        M.ss=ss
        ML2=arange(len(ML))*1.
        ML2=ML2/sum(ML2)
        M.ML2=ML2
        if show_plot:
            #plot
            col = cm.jet(linspace(0, 1, nt*nt))
            Nr=40000
            XX=random.rand(Nr)*width
            YY=random.rand(Nr)*height
            XX=concatenate((XX,zeros(500),ones(500)*width,linspace(0,width,500),linspace(0,width,500)))
            YY=concatenate((YY,linspace(0,height,500),linspace(0,height,500),ones(500)*height,zeros(500)))
            TT=ones(len(XX))
            RR=array([XX,YY,TT,TT]).T
            M.evaluate_map(inputX=RR,inputY=zeros(len(RR)))
            figt=plt.figure(2, frameon=False)
            figt.set_size_inches(5, 5)
            ax1=plt.Axes(figt, [0., 0., 1., 1.])
            ax1.set_axis_off()
            figt.add_axes(ax1)


            for i in xrange(nt*nt):
                if M.ivals.has_key(ss[i]):
                    w=array(M.ivals[ss[i]])
                    DDD=array([XX[w],YY[w]]).T
                    if len(DDD)> 2:
                        ht=ConvexHull(DDD)
                        ax1.fill(*zip(*DDD[ht.vertices]), color=col[i], alpha=0.6)
                    #ax1.scatter(XX[w],YY[w],marker='o',edgecolor='none',color=col[i],s=50,alpha=0.2)
            cx=M.weights[0]
            cy=M.weights[1]
            ax1.plot(active[:,0],active[:,1],'k.')

            # compute Voronoi tesselation
            points2=array([cx,cy]).T
            vor = Voronoi(points2)
            pp=voronoi_plot_2d_local(vor,ax=ax1)
            ax1.set_xlim(0,width)
            ax1.set_ylim(0,height)
            #ax1.set_axis_off()
            plt.axis('off')
            plt.gca().set_axis_off()
            plt.gca().axes.get_xaxis().set_visible(False)
            plt.gca().axes.get_yaxis().set_visible(False)
            
            nnn='%05d' % jj
            figt.savefig(output_folder + '/plots/somplot/som_'+nnn+'.png',bbox_inches='tight',pad_inches=0)
            figt.clear()
            M.evaluate_map(inputX=DD,inputY=L)
        
    else:
        M=inM
        ML=M.ML
        ML2=M.ML2
        ss=M.ss
    while True:
        t=random.choice(len(ML2), 1, p=ML2)[0]
        if (M.ivals.has_key(ss[t])):
            break
    cell=ss[t]

    
    while True:
        keep=True
        xt=random.normal(mean(active[M.ivals[cell],0]),max([std(active[M.ivals[cell],0]),0.01]))
        yt=random.normal(mean(active[M.ivals[cell],1]),max([std(active[M.ivals[cell],1]),0.01]))
        at=random.normal(mean(active[M.ivals[cell],2]),max([std(active[M.ivals[cell],2]),0.01]))
        rt=random.normal(mean(active[M.ivals[cell],3]),max([std(active[M.ivals[cell],3]),0.01]))
        if (xt < 0) or (xt>width) : keep=False
        if (yt < 0) or (yt>height) : keep=False
        if (at < amp_min)  or (at > amp_max) : keep=False
        if (rt < rad_min) or (rt > rad_max): keep=False

        if keep:
            new=array([xt,yt,at,rt])
            newL,neval=lnlike(new,data,nlog=neval)
            if newL > LLog_min: break
    return [M,new,neval]

parser = SafeConfigParser()
parser.read("../config.ini")
#again, it would be best to replace this relative path with something better

prefix = parser.get("Misc", "prefix")
location = parser.get("Misc", "location")
output_folder = location + "/" + prefix 
image_location = output_folder + "/" + prefix + "_noised.npy"
no_noise_location = output_folder + "/" + prefix + "_clean.npy"

#image parameters
width = int(parser.get("Sampling","width"))
height = int(parser.get("Sampling","height"))

#sampling parameters
noise_lvl = float(parser.get("Sampling", "noise"))
amp_min = float(parser.get("Sampling", "amp_min"))
amp_max = float(parser.get("Sampling", "amp_max"))

rad_min = float(parser.get("Sampling", "rad_min"))
rad_max = float(parser.get("Sampling", "rad_max"))

niter = int(parser.get("Sampling", "niter"))
num_active_points = int(parser.get("Sampling", "num_active"))
num_som_iter = int(parser.get("Sampling", "num_som_iter"))
num_som_points = int(parser.get("Sampling", "num_som_points"))

#output parameters
output_filename = prefix + "_" + parser.get("Output", "output_filename")

show_plot = parser.getboolean("Output", "plot")

#detection parameters
neighbor_dist = float(parser.get("Detection", "neighbor_dist"))
cutoff = int(parser.get("Detection", "cutoff"))
detected_processed_filename = prefix + "_processed_" + parser.get("Detection", "detected_filename")
detected_all_filename = prefix + "_all_" + parser.get("Detection", "detected_filename")


"""
#legacy code
SrcArray = [[43.71, 22.91, 10.54, 3.34],
            [101.62, 40.60, 1.37, 3.40],
            [92.63, 110.56, 1.81, 3.66],
            [183.60, 85.90, 1.23, 5.06],
            [34.12, 162.54, 1.95, 6.02],
            [153.87, 169.18, 1.06, 6.61],
            [155.54, 32.14, 1.46, 4.05],
            [130.56, 183.48, 1.63, 4.11]]

data_or=make_source(src_array = SrcArray,height=height, width=width)
noise = noise_lvl
data=add_gaussian_noise(mean=0,sd=noise,data=data_or)
"""

#more legacy code
#filee='../bayes-detect-master/simulated_images/multinest_toy_noised'
#s=open(filee,'r')
#data=pickle.load(s)
#s.close()

#sys.exit(0)

os.system('mkdir -p ' + output_folder + '/plots/')
os.system('mkdir -p ' + output_folder + '/plots/detected')
os.system('mkdir -p ' + output_folder + '/plots/6plot')
os.system('mkdir -p ' + output_folder + '/plots/4plot')
os.system('mkdir -p ' + output_folder + '/plots/somplot')
os.system('mkdir -p ' + output_folder + '/plots/3dplot')

data = load(image_location)
data_or = load(no_noise_location)


#more legacy code
#Im=pf.open('ufig_20_g_sub_500_sub_small.fits')
#data=Im[0].data
#Im.close()

x=arange(width,dtype='float')
y=arange(height,dtype='float')
xx,yy=meshgrid(x,y)


start = timeit.default_timer()
neval=0
Np=num_active_points
AC=zeros((Np,5))
#initially active points is a array of 0s

AC[:,0]=random.rand(Np)*(width - 1.)
AC[:,1]=random.rand(Np)*(height - 1.)
AC[:,2]=random.rand(Np)*(amp_max-amp_min) + amp_min
AC[:,3]=random.rand(Np)*(rad_max-rad_min) + rad_min
#seed the active points with uniform random numbers in the wanted range

for i in xrange(Np):
    AC[i,4],neval=lnlike(AC[i,0:4],data,nlog=neval)
    #compute the log likelihood for each of those points


print 'done with active'

Niter=niter
points=zeros((Niter,5))

detected = zeros((1, 6)) #remember to ignore the first item
detected_count = [-1]

l = 0 #number of clusters got after each iteration index
clusterCount = zeros(((Niter/num_som_iter)+1,2))  #list to keep the count of number of clusters after each iteration

for i in xrange(Niter):
    if i%num_som_iter == 0:
        print i
    reject=argmin(AC[:,4])
    minL=AC[reject,4]
    if i%num_som_iter == 0:
        Map,new,neval=sample_som(i,AC,neval,minL,nt=4,nit=150,create='yes',sample='yes')
        #create=yes -> make a new som
        count = make_plot(points,AC,i)
        #print count, highestClusterCount
        if(count < highestClusterCount["count"]): break
	
        clusterCount[l][1] = count
        clusterCount[l][0] = i
        l = l+1
        make_3dplot(AC,i)
        
	#plot things with make_plot
        #sources, detected, detected_count = detect_sources(AC, i, detected, detected_count, neighbor_dist)
        #run detection on the active points
        #look_at_results(sources, i)
        #plot the detected items
    else:
        Map,new,neval=sample_som(i,AC,neval,minL,nt=4,nit=150,create='no',sample='yes',inM=Map)
        #sample from the som w/o creating a new one
#legacy code
    #while True:
      #  new=sample()
#        newL,neval=lnlike(new,data,nlog=neval)
#        if newL > minL:
#            break
    newL,neval=lnlike(new,data,nlog=neval)
    points[i]=AC[reject]
    AC[reject,0:4]=new
    AC[reject,4]=newL
    
print clusterCount

#make_clusterCountPlot(clusterCount)

stop = timeit.default_timer()

#write some stats to a text file
with open(output_folder + "/stats.txt", "wb") as f:
    f.writelines(["seconds,%d\n"%(stop - start), "Log evaluations,%d"%neval])

print stop - start, 'seconds'
print neval, 'Log evaluations'

#savetxt('out_points_som.txt',points,fmt='%.6f')
savetxt(output_folder + "/" + output_filename, points,fmt='%.6f')

#deal with the cutoff
dc = array(detected_count)
d = c_[detected, dc] #concatenate them with dc as the last column
d = d[d[:,-1].argsort()] #sort it by detected count (the last column). this is actually not necessary
above_cutoff = where(d[:,-1] >= cutoff)[0]
filtered_detected = d[above_cutoff]


header = "x,y,a,r,L,detection_count"
savetxt(output_folder + "/" + detected_processed_filename, filtered_detected, fmt="%.6f", delimiter=",",header=header) 
print "wrote to file: " + output_folder + "/" + detected_processed_filename
savetxt(output_folder + "/" + detected_all_filename, d, fmt="%.6f", delimiter=",",header=header) 
print "wrote to file: " + output_folder + "/" + detected_all_filename
