import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from TSClass_function import *
from TSClass_lorenz import *


#%% Functions

def show_result(y, y_pred, full=False):
    plt.plot(y[150:750], c='red', lw=0.5)
    plt.plot(y_pred[150:750], c='blue')
    plt.show()

    if full == True:
        plt.plot(y, c='red')
        plt.plot(y_pred, c='blue')
        plt.show()

def function_example_ma(start_idx, end_idx, dt, sigma):
    """
    Example of callable object written the "correct way",
    that accepts as first arguments range of indexes [start_idx, end_idx)
    and return an array of (end_idx-start_idx,) values.
    In this case it is sin + noise
    """
    x1=start_idx*dt
    x2=(end_idx-1)*dt
    m=12*dt
    x=np.linspace(x1,x2,end_idx - start_idx)
    #assert (np.shape(x)[0]==)
    return 100+6*np.sin(2*np.pi*x/m)+moving_avarage(start_idx, end_idx,dt,sigma)

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

#%% Constants
inpt_window = 100
outp_window = 1
n=30000
dt=0.1

#%% Create Data

tsc1 = TSC_function(savevals=True)
tsc2 = TSC_function(savevals=True)


serie_train  = tsc1.get_next_N_vals(n,function_example_ma, dt, sigma=2.5)
serie_test= tsc2.get_next_N_vals(n,function_example_ma, dt, sigma=2.5)


#Scale Series
MMscaler = MinMaxScaler(feature_range=(-1, 1))
serie_train = MMscaler.fit_transform(serie_train.reshape(-1,1)).flatten()
serie_test = MMscaler.fit_transform(serie_test.reshape(-1,1)).flatten()

#%% Forecasting
from noHW_LSTM import *

print("Initializing noHW_LSTM object")
model = noHW_LSTM(inpt_window,outp_window)

#Create test/train sets
X_test,y_test=model.sequence_splitter(serie_train)
X_train,y_train = model.sequence_splitter(serie_test)

#Build model
NN_model = model.build_NN()

#Fit model
history,y_pred,y_true,MSE=model.fit_NN(X_train,y_train,X_test,y_test,epochs=5,bsize=64)

#Show result
print("Showing results")
show_result(y_pred, y_true)


