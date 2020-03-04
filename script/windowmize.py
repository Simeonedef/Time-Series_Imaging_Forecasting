# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 16:15:43 2019

@author: 39389
"""

"""
Input:
 x: array of shape (N,)
Output
 X: array of shape ()
"""
import numpy as np
def windowmize(x,window_size=2, stride = 1):
#    N=np.shape(x)(0)
#    X=np.array(x[N-window_size,N]).reshape(1,-1)
#    
#    for i in range(1,N):
#        X=np.r_['0', X, x[N-i*stride-window_size,N-i*stride].reshape(1,-1)]
#        
#    return X
    N=np.shape(x)[0]
    #n_windows=(N-window_size)//stride+1
    #lag=(N-window_size)%stride
    [n_windows,lag] = windomize_size(N, window_size, stride)
    
    windows=np.zeros([n_windows,window_size])
    for i in range(n_windows):
        shift=i*stride
        windows[i,:]=x[lag+shift:lag+window_size+shift]
    return windows
    #print(f"windows: {windows}")

"""
Input:
 x_N: size of array of shape (N,)
 
Output
 X: array of shape ()
"""
def windomize_size(x_N, window_size, stride):
    N=x_N
    n_windows=(N-window_size)//stride+1
    lag=(N-window_size)%stride
    return [n_windows,lag]
    