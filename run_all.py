
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

def is_hit(data1, data2):
    distance = ((data1[0] - data2[0])**2 + (data1[1] - data2[1])**2)**0.5
    if distance < (data1[3] + data2[3]):
        return 1
    else:
        return 0

num_runs = 1
for i in range(num_runs):
    # In[2]:

    start = timeit.default_timer()
    prefix = 4


    # In[5]:

    #read config.ini template
    parser = SafeConfigParser()
    #todo: do something about not using the relative path
    parser.read("config.ini")

    output_dir = parser.get("Misc", "location") + "/" + str(prefix)
    os.system('mkdir -p ' + output_dir)


    # In[6]:

    #change prefix and write
    parser.set("Misc","prefix",str(prefix))
    parser.set("Output", "plot","False")
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
    #ig.run(fileout)

    # In[6]:
    #ns.run(fileout)

    # In[7]:
    ps.run(fileout)

    # In[8]:
    stop = timeit.default_timer()
    print stop - start, 'seconds'
    
    originalData = load(output_folder +"/" + prefix + "_srcs.npy")
    finalData = loadtxt(output_folder +"/" + prefix + "_finalData.txt")
    #print finalData
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    i = 0
    while i < len(finalData):
        j = 0
        while j < len(originalData):
            a = is_hit(finalData[i],originalData[j])
            if(a == 1):
                tp = tp + 1
                finalData = delete(finalData,i,0)
                originalData = delete(originalData,j,0)
                i = 0
                j = 0
                break
            j = j + 1
        if j == len(originalData):
            i = i + 1
    fp = len(finalData)
    print "data",tp,fp
