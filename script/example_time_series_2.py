# -*- coding: utf-8 -*-
"""
Created on Sat Jan  4 19:49:28 2020

@author: 39389
"""
import matplotlib.pyplot as plt
from holt_winter import *
from TSClass_function import *
from TSClass_lorenz import *
import math
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from Conv_LSTM import *
import gc
#%% moving avarage 

def moving_avarage(start_idx, end_idx,dt,sigma):
    alpha=0.35
    beta=0.15
    gamma=0.10
    delta=0.35
    eta=-0.03
    omega=0.05
    
    ma=np.zeros(end_idx - start_idx)
    ma[0:6]=np.random.normal(loc=0.0, scale=sigma, size=6)
    for i in range(4,end_idx - start_idx):
        ma[i]=alpha*ma[i-1]+beta*ma[i-2]+gamma*ma[i-3]+delta*ma[i-4]+eta*ma[i-5]+ \
        omega*ma[i-6]+np.random.normal(loc=0.0, scale=sigma, size=1)
    return ma




#%%
def function_example_ma(start_idx, end_idx, dt, sigma):
    """
    Example of callable object written the "correct way",
    that accepts as first arguments range of indexes [start_idx, end_idx)
    and return an array of (end_idx-start_idx,) values.
    In this case it is sin + noise
    """
    x1=start_idx*dt
    x2=(end_idx-1)*dt
    m=28*dt
    x=np.linspace(x1,x2,end_idx - start_idx)
    return 700+(160*np.sin(2*np.pi*x/(67.345*m))+30*np.cos(2*np.pi*x/(3.4*m)))* \
           (1+0.3*np.sin(2*np.pi*x/m))+moving_avarage(start_idx, end_idx,dt,sigma)
#%%
n=200000
dt=0.1

"""use h=12 as prediction horizon, m=28"""

np.random.seed(57)
tsc1 = TSC_function(savevals=True)
tsc2 = TSC_function(savevals=True)

serie  = tsc1.get_next_N_vals(n,function_example_ma, dt, sigma=2.8)
serie_test= tsc2.get_next_N_vals(n,function_example_ma, dt, sigma=2.8)
#%%
plt.plot(serie)
#%%
plt.plot(serie[1000:2000])
#%%%
plt.plot(serie[4000:4100])