# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 17:37:48 2019

@author: 39389
"""

import numpy as np
import math
x=np.array([3,4,6,3,8,12])
x=np.array(range(13))
math.ceil(3.7)


window_size=5
stride = 3

N=np.shape(x)[0]
n_windows=(N-window_size)//stride+1
lag=(N-window_size)%stride

windows=np.zeros([n_windows,window_size])


for i in range(n_windows):
    shift=i*stride
    print(windows[i,:])
    print(x[lag-1+shift:lag+window_size+shift])
    
    
    print(np.size( windows[i,:]))
    print(np.size( x[lag+shift:lag+window_size+shift]))
    
    windows[i,:]=x[lag+shift:lag+window_size+shift]
print(f"windows: {windows}")
