import argparse
import numpy as np
from numpy.random import uniform, randint
import test_image
from ConfigParser import SafeConfigParser
import os

def run(configfile):
    parser = SafeConfigParser()
    #todo: do something about not using the relative path
    parser.read(configfile)

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
    

    #make a src array [[x,y,a,r], ...]
    src_array = np.zeros((num_sources, 4))
    for source in range(src_array.shape[0]):
        src_array[source, 0] = randint(width)
        src_array[source, 1] = randint(height)
        src_array[source, 2] = uniform(low = amp_min, high = amp_max)
        src_array[source, 3] = uniform(low = rad_min, high = rad_max)

    #we first make a non noisy image with make_source
    non_noisy_sources = test_image.make_source(src_array, height, width)

    #then noise is added (obvious from function name)
    noised = test_image.add_gaussian_noise(0, noise, non_noisy_sources)

    #these are all stored as numpy binary file
    #http://docs.scipy.org/doc/numpy/reference/generated/numpy.save.html
    normal = output_dir + "/" + prefix + "_noised.npy"
    with open(normal, "wb") as f:
        np.save(f, noised)

    clean = output_dir + "/" + prefix + "_clean.npy"
    with open(clean, "wb") as f:
        np.save(f, non_noisy_sources)

    srcs = output_dir + "/" + prefix + "_srcs.npy"
    with open(srcs, "wb") as f:
        np.save(f, src_array)
