import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from holt_winter import *
from TSClass_function import *
from TSClass_lorenz import *
#%% Functions

def show_result(y, y_pred, full=False):
    plt.plot(y[150:750], c='red', lw=0.7)
    plt.plot(y_pred[150:750], c='blue',ls='--')
    plt.show()

    if full == True:
        plt.plot(y, c='red')
        plt.plot(y_pred, c='blue')
        plt.show()


#%% Constants
img_size = 32


#%% Create Data

#Create Data
tsc3 = TSC_lorenz(savevals=True)
n=30000
dt=0.1

serie  = 100+tsc3.get_next_N_vals(n)
serie_test=100+tsc3.get_next_N_vals(n)

#%% Create Holt Winter object
print("Initializing Holt Winter")
HW=Holt_Winters_NN(serie,serie_test,m=1,h=2,windowsize=img_size,stride=1,alpha=0.25,beta=0,gamma=0.35,pr=1,compute_mtf=False)

#%% Call NN
print("Initializing Conv_MLP object")
from Conv_MLP import *
model = Conv_MLP(img_size=img_size,N_Channel=2) #if compute_mtf=False -> set N_Channel=2


#%% Fit model
print("Starting Conv_MLP training")
history,y_pred,y_true,MSE = model.train_HW(HW,epochs=5, bsize=32)
# model.save("CONV_LSTM.hdf5")
print("Finished training")
print(MSE)


print("Showing results")
show_result(y_pred, y_true)
