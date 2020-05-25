import matplotlib.pyplot as plt
import numpy as np
from utils import show_result
from TSClass_function import *
from TSClass_lorenz import *
from noHW_LSTM import *


#%% Functions
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


#You can replace this below with wherever you want the predictions csv file to be
preds_filepath = "predictions_test"

print("Initializing LSTM object")
model = noHW_LSTM(inp_window=inpt_window,out_window=outp_window)
print("Starting LSTM training")
history,y_pred,y_true,MSE = model.fit_NN(serie_train[0:n],serie_test[0:n], epochs=2, bsize=64, p_filepath=preds_filepath, callbacks=False)

show_result(y_true, y_pred, full=False, title="LSTM without HW")


