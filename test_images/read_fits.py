import pyfits as pf
import matplotlib.pyplot as plt
import numpy as np

H=pf.open('test6_i.fits')


xx,yy=np.loadtxt('test6_objects.txt',unpack=True, usecols=(0,1))

Image=H[0].data
#Note there one very bright object, so we reduce the range from 0 to ~60
plt.imshow(Image, vmin=0, vmax=Image.max()*0.001)
plt.scatter(xx,yy, s=200, facecolors='none', edgecolors='k',lw=2)

plt.xlim(0,400)
plt.ylim(0,400)
plt.show()
