
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


start = timeit.default_timer()

newprefix = 3

#read config.ini template
parser = SafeConfigParser()
parser.read("config.ini")

#image parameters
width = int(parser.get("Image","width"))
height = int(parser.get("Image","height"))
noise_lvl = float(parser.get("Image", "noise"))
amp_min = float(parser.get("Image", "amp_min"))
amp_max = float(parser.get("Image", "amp_max"))

rad_min = float(parser.get("Image", "rad_min"))
rad_max = float(parser.get("Image", "rad_max"))
num_sources = int(parser.get("Image", "num_items"))
noise = float(parser.get("Image", "noise"))

prefix = parser.get("Misc", "prefix")
output_dir = parser.get("Misc", "location") + "/" + prefix

#change prefix and write
parser.set("Misc","prefix",str(newprefix))
fileout='config'+'_'+str(newprefix)+'.ini'
F=open(fileout,'w')
parser.write(F)
F.close()

#ig.run(fileout)

#ns.run(fileout)

ps.run(fileout)

stop = timeit.default_timer()
print stop - start, 'seconds'



