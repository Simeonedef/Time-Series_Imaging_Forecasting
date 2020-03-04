# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 17:13:30 2019

@author: Stefano d'Apolito
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
import copy



#TO SEE HOW TO USE HOLT-WINTER JUST GO TO LINE 124 (first part is about producing time series and defining tf model)




#%% please use function_example_ma (moving avarege noise)
def function_example(start_idx, end_idx, dt, sigma):
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
    #assert (np.shape(x)[0]==)
    return 100+(np.sqrt(x)+25*np.cos(2*np.pi*x/(3.4*m)))*(1+0.07*np.sin(2*np.pi*x/m))+np.random.normal(loc=0.0, scale=sigma, size=(end_idx - start_idx,))

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
        ma[i]=alpha*ma[i-1]+beta*ma[i-2]+gamma*ma[i-3]+delta*ma[i-4]+eta*ma[i-5]+omega*ma[i-6]+np.random.normal(loc=0.0, scale=sigma, size=1)
    return ma

#%% like funcion example but with moving avarege instead of normal noise
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
    #assert (np.shape(x)[0]==)
    return 800+(160*np.sin(2*np.pi*x/(107.345*m))+30*np.cos(2*np.pi*x/(3.4*m)))*(1+0.3*np.sin(2*np.pi*x/m))+np.random.normal(loc=0.0, scale=sigma, size=(end_idx - start_idx,))+moving_avarage(start_idx, end_idx,dt,sigma)

#%% example of how to build a tf  model, not relevant to you
def build_model(w_size):
  model = keras.Sequential([
    #24 equal to the window size
    tf.keras.layers.Flatten(input_shape=(w_size,w_size)),      
    layers.Dense(50, activation='relu'),
    layers.Dense(50, activation='relu'),
    layers.Dense(1)
  ])

  optimizer = tf.keras.optimizers.RMSprop(0.001)

  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse'])
  return model

#%% example of how to build a tf convolutional  model, not relevant to you
def build_model_conv(w_size):
  model = keras.Sequential([
    #24 equal to the window size 
    layers.Conv2D(1, (5, 5), activation='relu', input_shape=(w_size, w_size,1)),
    layers.Conv2D(1, (4, 4), activation='relu', input_shape=(w_size, w_size,1)),
    tf.keras.layers.Flatten(input_shape=(1,28,28)),
    layers.Dense(10, activation='relu'),
    layers.Dense(7, activation='relu'),
    layers.Dense(1)
  ])

  optimizer = tf.keras.optimizers.RMSprop(0.001)

  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse'])
  return model
#%%
#prepere test and training serie   
  
data=np.load("../data/dowjones_86_19.npy")

serie=data[0:4000]

serie_test=data[4000:8000]

"""
tsc1 = TSC_function(savevals=True)
tsc2 = TSC_function(savevals=True)
#tsc3 = TSC_lorenz(savevals=True)
n=200000
dt=0.1
serie  = tsc1.get_next_N_vals(n,function_example_ma, dt, sigma=2.6)
serie_test= tsc2.get_next_N_vals(n,function_example_ma, dt, sigma=2.6)

#serie  = 100+tsc1.get_next_N_vals(n)
#serie_test=100+tsc2.get_next_N_vals(n)
print(np.shape(tsc1.vals))
"""
#%%
#tsc3.plot_range()
#%%
plt.plot(serie[100:800]) 

#%%create holt winter object

#note: 1) serie and series_est must have the same length (the implementaton was easier), 
#      2) m is the seasonality. for Lorenz (that has not seasonality put m=1), otherwise put the hardcoded value (without dt) of line 64
#      3) h is the horizon of the prediction (IMPORTANT: if m!=1,  it must be m<h)
#      4) pr indicate the number of iteration to estimate the best parameter for HW. if m=1, pr=1 is enough, otherwise use pr=3,4
#      5) pyts does not manage to compute the normalised mtf immage for Lorenz, so put compute_mtf=false in this case

HW=Holt_Winters_NN(serie,serie_test,m=1,h=1,windowsize=10,stride=1,alpha=0.25,beta=0,gamma=0.35,pr=2,compute_mtf=False)
#%% pure HW
HWm=Holt_Winters_multiplicative(serie,serie_test,m=1,h=1,alpha=0.25,beta=0,gamma=0.35)
HWm.compute_states()
HWm.parameter_refinment()
HWm._windowsize=10
HWm._stride=1
test_predictions_HWm=HWm.compute_states_test()

#%%
#create NN model

model = build_model_conv(10)
model.summary()



#Probably is more carefull pass all the list e.g. HW gadf trhough a copy.deepcopy
model.fit(copy.deepcopy(np.expand_dims(HW.gadf,axis=3)),HW.obtain_training_output(), epochs=5)

#%%predict
test_predictions_tmp= model.predict(copy.deepcopy(np.expand_dims(HW.gadf_test,axis=3))).flatten()
test_predictions= test_predictions_tmp*HW.forecast_multiplier;
#%%
MSE_test=((HW.test_output-test_predictions)**2).mean()
#%%
MSE_HW=((HW.test_output-test_predictions_HWm)**2).mean()

#%%
plt.figure
plt.plot(test_predictions[1500:1600],'r-')
plt.plot(HW.test_output[1500:1600],'b-')  
plt.plot(test_predictions_HWm[1500:1600],'g-')
#%%
plt.figure
plt.plot(test_predictions[0:3000],'r-')
plt.plot(HW.test_output[0:3000],'b-')  
plt.plot(test_predictions_HWm[0:3000],'g-')
#%%
plt.figure
plt.plot(test_predictions[1800:1900],'r-')
plt.plot(HW.test_output[1800:1900],'b-')  
plt.plot(test_predictions_HWm[1800:1900],'g-')

#%%
plt.plot(test_predictions[91980:92670],'r-')
plt.plot(HW.test_output[91980:92670],'b-')
plt.plot(test_predictions_HWm[91980:92670],'g-')

#%%
plt.plot(serie[1800:1900]-serie_test[1800:1900],'b-')
#%%
HW.obtain_training_output().mean()



