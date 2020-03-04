# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 15:21:52 2019

@author: 39389
"""
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from pyts.image import GramianAngularField
from pyts.datasets import load_gunpoint

import numpy as np
import matplotlib.pyplot as plt
#import copy
import pyts
from windowmize import *

# Parameters
#X, _, _, _ = load_gunpoint(return_X_y=True)
#X=X[:].reshape(1,-1)

serie=np.array([112, 118, 132, 129, 121, 135, 148, 148, 136, 119, 104, 118, 
                115, 126, 141, 135, 125, 149, 170, 170, 158, 133, 114, 140, 
                145, 150, 178, 163, 172, 178, 199, 199, 184, 162, 146, 166, 
                171, 180, 193, 181, 183, 218, 230, 242, 209, 191, 172, 194, 
                196, 196, 236, 235, 229, 243, 264, 272, 237, 211, 180, 201, 
                204, 188, 235, 227, 234, 264, 302, 293, 259, 229, 203, 229, 
                242, 233, 267, 269, 270, 315, 364, 347, 312, 274, 237, 278, 
                284, 277, 317, 313, 318, 374, 413, 405, 355, 306, 271, 306, 
                315, 301, 356, 348, 355, 422, 465, 467, 404, 347, 305, 336, 
                340, 318, 362, 348, 363, 435, 491, 505, 404, 359, 310, 337, 
                360, 342, 406, 396, 420, 472, 548, 559, 463, 407, 362, 405, 
                417, 391, 419, 461, 472, 535, 622, 606, 508, 461, 390, 432])

minx,maxx=min(serie),max(serie)
r1,r2 = (-1,1)
serie1=(r2-r1)*(serie-minx)/(maxx-minx)+r1
X=serie1.reshape(1,-1)

# Transform the time series into Gramian Angular Fields
gasf = GramianAngularField(image_size=24, method='summation',sample_range=None)
X_gasf = gasf.fit_transform(X)
gadf = GramianAngularField(image_size=24, method='difference',sample_range=None)
X_gadf = gadf.fit_transform(X)

# Show the images for the first time series
fig = plt.figure(figsize=(12, 7))
grid = ImageGrid(fig, 111,
                 nrows_ncols=(1, 2),
                 axes_pad=0.15,
                 share_all=True,
                 cbar_location="right",
                 cbar_mode="single",
                 cbar_size="7%",
                 cbar_pad=0.3,
                 )
images = [X_gasf[0], X_gadf[0]]
titles = ['Gramian Angular Summation Field',
          'Gramian Angular Difference Field']
for image, title, ax in zip(images, titles, grid):
    im = ax.imshow(image, cmap='rainbow', origin='lower')
    ax.set_title(title, fontdict={'fontsize': 16})
ax.cax.colorbar(im)
ax.cax.toggle_label(True)


#%%


serie=np.array([112, 118, 132, 129, 121, 135, 148, 148, 136, 119, 104, 118, 
                115, 126, 141, 135, 125, 149, 170, 170, 158, 133, 114, 140, 
                145, 150, 178, 163, 172, 178, 199, 199, 184, 162, 146, 166, 
                171, 180, 193, 181, 183, 218, 230, 242, 209, 191, 172, 194, 
                196, 196, 236, 235, 229, 243, 264, 272, 237, 211, 180, 201, 
                204, 188, 235, 227, 234, 264, 302, 293, 259, 229, 203, 229, 
                242, 233, 267, 269, 270, 315, 364, 347, 312, 274, 237, 278, 
                284, 277, 317, 313, 318, 374, 413, 405, 355, 306, 271, 306, 
                315, 301, 356, 348, 355, 422, 465, 467, 404, 347, 305, 336, 
                340, 318, 362, 348, 363, 435, 491, 505, 404, 359, 310, 337, 
                360, 342, 406, 396, 420, 472, 548, 559, 463, 407, 362, 405, 
                417, 391, 419, 461, 472, 535, 622, 606, 508, 461, 390, 432])


minx,maxx=min(serie),max(serie)
r1,r2 = (-1,1)
serie1=(r2-r1)*(serie-minx)/(maxx-minx)+r1
#X=serie1.reshape(1,-1)
X=serie1

Xw = windowmize(X,12,1)

#%%
# Transform the time series into Gramian Angular Fields
print("starting")
gasf = GramianAngularField(image_size=1., method='summation',sample_range=None)
X_gasf = gasf.fit_transform(Xw)
gadf = GramianAngularField(image_size=1., method='difference',sample_range=None)
X_gadf = gadf.fit_transform(Xw)
print("finished")
## Show the images for the first time series
#for l in range(0,133):
#    fig = plt.figure(figsize=(12, 7))
#    grid = ImageGrid(fig, 111,
#                     nrows_ncols=(1, 2),
#                     axes_pad=0.15,
#                     share_all=True,
#                     cbar_location="right",
#                     cbar_mode="single",
#                     cbar_size="7%",
#                     cbar_pad=0.3,
#                     )
#    images = [X_gasf[l], X_gadf[l]]
#    titles = ['Gramian Angular Summation Field',
#              'Gramian Angular Difference Field']
#    for image, title, ax in zip(images, titles, grid):
#        im = ax.imshow(image, cmap='rainbow', origin='lower')
#        ax.set_title(title, fontdict={'fontsize': 16})
#    ax.cax.colorbar(im)
#    ax.cax.toggle_label(True)
