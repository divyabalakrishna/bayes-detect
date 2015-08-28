
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


'''num_runs = 3
for i in range(num_runs):
    # In[2]:

    start = timeit.default_timer()
    newprefix = i


    # In[5]:

    #read config.ini template
    parser = SafeConfigParser()
    #todo: do something about not using the relative path
    parser.read("config.ini")

    prefix = newprefix
    output_dir = parser.get("Misc", "location") + "/" + str(prefix)
    os.system('mkdir -p ' + output_dir)


    # In[6]:

    #change prefix and write
    parser.set("Misc","prefix",str(newprefix))
    fileout="files"+"/"+str(newprefix)+"/"+'config'+'_'+str(newprefix)+'.ini'
    #fileout='config'+'_'+str(newprefix)+'.ini'
    F=open(fileout,'w')
    parser.write(F)
    F.close()


    # In[15]:
    ig.run(fileout)

    # In[6]:
    ns.run(fileout)

    # In[7]:
    ps.run(fileout)

    # In[8]:
    stop = timeit.default_timer()
    print stop - start, 'seconds'
'''
fileout="files/2/config_2.ini"
ps.run(fileout)


