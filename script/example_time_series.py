# -*- coding: utf-8 -*-
"""
Created on Sat Jan  4 19:49:28 2020

@author: 39389
"""
import matplotlib.pyplot as plt
from holt_winter import *
from TSClass_function import *
from TSClass_lorenz import *
import numpy as np
import argparse
from utils import show_result

from Conv_LSTM import *
from Conv_MLP import *
from noHW_LSTM import *

parser = argparse.ArgumentParser(description="model")
parser.add_argument("--model", help="Conv_MLP, Conv_LSTM, or LSTM", type=str)
parser.add_argument("--hw", help="set it to run HW, no HW for LSTM", action="store_true")
parser.add_argument("--no_cb", help = "set it to not record weights, losses, only predictions at the end", action="store_true")
args = parser.parse_args()
model_choice = args.model
hw_choice = args.hw
no_cb = args.no_cb

if model_choice is None:
    print("Please choose a model: Conv_MLP, Conv_LSTM, or LSTM")
    print('In the following manner: --model="model_name"')
    exit()

if model_choice != "Conv_MLP" and model_choice != "Conv_LSTM" and model_choice != "LSTM":
    print("Please supply a valid model name: Conv_MLP, Conv_LSTM, or LSTM")
    exit()

def moving_average(start_idx, end_idx, dt, sigma):
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
    return 700 + (160*np.sin(2*np.pi*x/(67.345*m))+30*np.cos(2*np.pi*x/(3.4*m))) * \
           (1+0.3*np.sin(2*np.pi*x/m)) + moving_average(start_idx, end_idx, dt, sigma)
#%%

print("##############################")
print("####### Noisy Function #######")
print("#### Training and testing ####")
print("##############################")

#================= Constants ===============
n=200000
dt=0.1
img_size = 40
preds_filepath="predictions_noisy"
w_filepath = "weights_noisy"
l_filepath ="logs_noisy"
seq_len = 3
out_window = 1
test_size=10000
h=12
epochs = 10
bsize = 32
in_window = 100
if no_cb:
    print("=========================================")
    print("No callbacks")
    print("=========================================")
    callbacks = False
else:
    print("=========================================")
    print("Logs will be in: ", l_filepath)
    print("Weights will be in: ", w_filepath)
    print("Predictions will be in: ", preds_filepath)
    print("=========================================")
    callbacks = True
#================= Constants END ===============

"""use h=12 as prediction horizon, m=28"""

np.random.seed(57)
tsc1 = TSC_function(savevals=True)
tsc2 = TSC_function(savevals=True)

serie  = tsc1.get_next_N_vals(n,function_example_ma, dt, sigma=2.8)
serie_test= tsc2.get_next_N_vals(n,function_example_ma, dt, sigma=2.8)

"""
Plots of the series
"""
# %%
# plt.plot(serie)
# plt.show()
# #%%
# plt.plot(serie[1000:2000])
# plt.show()
# %%%
# plt.plot(serie[4000:4100])
# plt.show()

"""
HW Object initialization
"""
if hw_choice:
    print("Initializing Holt Winter")
    HW=Holt_Winters_NN(serie,serie_test,m=28,h=h,windowsize=img_size,stride=1,alpha=0.35,beta=0.15,gamma=0.1)

"""
Conv LSTM with HW
"""
if model_choice == "Conv_LSTM" and hw_choice:
    print("Initializing Conv_LSTM object")
    Conv_LSTM_HW = Conv_LSTM(img_size, seq_len, in_window=in_window, out_window=out_window, conv_layers=0, lstm_layers=2, dropout=0.4, pre_loaded=False, bidirectional=True, channels=2, test_size=test_size)
    print("Starting Conv_LSTM training")
    model_hw, y_true, y_pred = Conv_LSTM_HW.train_HW(HW, epochs=epochs, bsize=bsize, p_filepath=preds_filepath, l_filepath=l_filepath, w_filepath=w_filepath, callbacks=callbacks)


"""
Conv LSTM without HW
"""
if model_choice == "Conv_LSTM" and not hw_choice:
    print("Initializing Conv_LSTM object")
    Conv_LSTM_HW = Conv_LSTM(img_size, seq_len, in_window=in_window, out_window=out_window, conv_layers=0, lstm_layers=2, dropout=0.4, pre_loaded=False, bidirectional=True, channels=3, test_size=test_size)
    print("Starting Conv_LSTM training")
    model_hw, y_true, y_pred = Conv_LSTM_HW.train_series(serie[0:n],serie_test[0:test_size], epochs=epochs, bsize=bsize, p_filepath=preds_filepath, l_filepath=l_filepath, w_filepath=w_filepath, h=h, callbacks=callbacks)

"""
Conv MLP with HW
"""
if model_choice == "Conv_MLP" and hw_choice:
    print("Initializing Conv_MLP object")
    model = Conv_MLP(img_size=img_size,N_Channel=2, test_size=test_size) #if compute_mtf=False -> set N_Channel=2
    print("Starting Conv_MLP training")
    history,y_pred,y_true,MSE = model.train_HW(HW, epochs=epochs, bsize=bsize, p_filepath=preds_filepath, l_filepath=l_filepath, w_filepath=w_filepath, callbacks=callbacks)

"""
Conv MLP without HW
"""
if model_choice == "Conv_MLP" and not hw_choice:
    print("Initializing Conv_MLP object")
    model = Conv_MLP(img_size=img_size,N_Channel=3, raw=True, test_size=test_size) #if compute_mtf=False -> set N_Channel=2
    print("Starting Conv_MLP training")
    history,y_pred,y_true,MSE = model.train_series(serie[0:n],serie_test[0:test_size], epochs=epochs, bsize=bsize, p_filepath=preds_filepath, l_filepath=l_filepath, w_filepath=w_filepath, h=h, callbacks=callbacks)

"""
Pure LSTM
"""
if model_choice == "LSTM":
    print("Initializing LSTM object")
    model = noHW_LSTM(inp_window=in_window,out_window=out_window)
    print("Starting LSTM training")
    history,y_pred,y_true,MSE = model.fit_NN(serie[0:n],serie_test[0:test_size], epochs=epochs, bsize=bsize, p_filepath=preds_filepath, l_filepath=l_filepath, w_filepath=w_filepath, h=h, callbacks=callbacks)

show_result(y_true, y_pred, full=False)

