
# coding: utf-8

# In[1]:

import argparse
from ConfigParser import SafeConfigParser
import os
from numpy import *
import image_generator.image_generator as ig
import model_nest_som.model_nest_som as ns
import post_sampling.post_sampling as ps
import timeit
import warnings
warnings.filterwarnings('ignore')

num_runs = 100
stats = zeros((num_runs,4))
for i in range(num_runs):
    # In[2]:
    start = timeit.default_timer()
    prefix = i
    num_active = 15000
    niter = 50001
    num_som_iter = 1000

    # In[5]:

    #read config.ini template
    parser = SafeConfigParser()
    parser.read("config.ini")

    output_dir = parser.get("Misc", "location") + "/" + str(prefix)
    os.system('mkdir -p ' + output_dir)


    # In[6]:

    #change prefix and write
    parser.set("Misc","prefix",str(prefix))
    parser.set("Output", "plot","True")
    parser.set("Sampling", "num_active",str(num_active))
    parser.set("Sampling", "niter",str(niter))
    parser.set("Sampling", "num_som_iter",str(num_som_iter))
    fileout="files"+"/"+str(prefix)+"/"+'config'+'_'+str(prefix)+'.ini'
    #fileout='config'+'_'+str(newprefix)+'.ini'
    F=open(fileout,'w')
    parser.write(F)
    F.close()

    parser.read(fileout)
    prefix = parser.get("Misc", "prefix")
    location = parser.get("Misc", "location")
    output_folder = location + "/" + prefix 

    # In[15]:
    ig.run(fileout)

    # In[6]:
    ns.run(fileout,1)

    # In[7]:
    tp,fp,ud = ps.run(fileout)

    # In[8]:
    stop = timeit.default_timer()
    print stop - start, 'seconds'
    
    
stats[i][0] = tp
stats[i][1] = fp
stats[i][2] = ud
stats[i][3] = stop - start
savetxt("files/" + str(prefix) + "/stats_new_15000.txt", stats,fmt='%.6f')
    