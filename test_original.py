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

def run(data, prefix):
    start = timeit.default_timer()
    num_active = 20000

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
    
    #chane width and height to image size
    print len(data)
    parser.set("Sampling", "width", str(len(data)))
    height = len(data[1:]) + 1
    parser.set("Sampling", "height", str(height))
    
    
    fileout="files"+"/"+str(prefix)+"/"+'config'+'_'+str(prefix)+'.ini'
    #fileout='config'+'_'+str(newprefix)+'.ini'
    F=open(fileout,'w')
    parser.write(F)
    F.close()

    parser.read(fileout)
    prefix = parser.get("Misc", "prefix")
    location = parser.get("Misc", "location")
    output_folder = location + "/" + prefix 

    normal = output_dir + "/" + prefix + "_noised.npy"
    with open(normal, "wb") as f:
        save(f, data)
        
    clean = output_dir + "/" + prefix + "_clean.npy"
    with open(clean, "wb") as f:
        save(f, data)
        
        
    # In[6]:
    ns.run(fileout)

    # In[7]:
    #tp,fp,ud = ps.run(fileout)

    # In[8]:
    stop = timeit.default_timer()
    print stop - start, 'seconds'