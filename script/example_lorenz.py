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

print("###############################")
print("## Lorenz Attractor Function ##")
print("#### Training and testing #####")
print("###############################")


# %%
n = 200000
dt = 0.1
img_size = 40
preds_filepath="predictions_lorenz"
w_filepath = "weights_lorenz"
l_filepath = "logs_lorenz"
seq_len = 3
out_window = 1
test_size = 10000
epochs = 10
bsize = 32
h = 10
in_window = 40
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

"""use h=10 as prediction horizon, m=1"""

np.random.seed(57)
tsc = TSC_lorenz(savevals=True)
serie  = 100+tsc.get_next_N_vals(n)
serie_test=100+tsc.get_next_N_vals(n)

"""
Plots of the series
"""
# # %%
# plt.plot(serie)
# plt.show()
# #%%
# plt.plot(serie[1000:2000])
# plt.show()
# # %%%
# plt.plot(serie[4000:4100])
# plt.show()

"""
HW Object initialization
"""
if hw_choice:
    print("Initializing Holt Winter")
    HW=Holt_Winters_NN(serie,serie_test,m=1,h=h,windowsize=img_size,stride=1,alpha=0.35,beta=0.15,gamma=0.1)

"""
Conv LSTM with HW
"""
if model_choice == "Conv_LSTM" and hw_choice:
    print("Initializing Conv_LSTM object")
    Conv_LSTM_HW = Conv_LSTM(img_size, seq_len, in_window=in_window, out_window=out_window, conv_layers=0, lstm_layers=2, dropout=0.4, pre_loaded=False, bidirectional=True, channels=2, test_size=test_size)
    print("Starting Conv_LSTM training")
    model_hw, test_real, test_predictions = Conv_LSTM_HW.train_HW(HW, epochs=epochs, bsize=bsize, p_filepath=preds_filepath, l_filepath=l_filepath, w_filepath=w_filepath, callbacks=callbacks)


"""
Conv LSTM without HW
"""
if model_choice == "Conv_LSTM" and not hw_choice:
    print("Initializing Conv_LSTM object")
    Conv_LSTM_HW = Conv_LSTM(img_size, seq_len, in_window=in_window, out_window=out_window, conv_layers=0, lstm_layers=2, dropout=0.4, pre_loaded=False, bidirectional=True, channels=3, test_size=test_size)
    print("Starting Conv_LSTM training")
    model_hw, test_real, test_predictions = Conv_LSTM_HW.train_series(serie[0:n],serie_test[0:test_size], epochs=epochs, bsize=bsize, p_filepath=preds_filepath, l_filepath=l_filepath, w_filepath=w_filepath, h=h, callbacks=callbacks)

"""
Conv MLP with HW
"""
if model_choice == "Conv_MLP" and hw_choice:
    print("Initializing Conv_MLP object")
    model = Conv_MLP(img_size=img_size,N_Channel=2, test_size=test_size) #if compute_mtf=False -> set N_Channel=2
    print("Starting Conv_MLP training")
    history,test_predictions,test_real,MSE = model.train_HW(HW, epochs=epochs, bsize=bsize, p_filepath=preds_filepath, l_filepath=l_filepath, w_filepath=w_filepath, callbacks=callbacks)

"""
Conv MLP without HW
"""
if model_choice == "Conv_MLP" and not hw_choice:
    print("Initializing Conv_MLP object")
    model = Conv_MLP(img_size=img_size,N_Channel=3, raw=True, test_size=test_size) #if compute_mtf=False -> set N_Channel=2
    print("Starting Conv_MLP training")
    history,test_predictions,test_real,MSE = model.train_series(serie[0:n],serie_test[0:test_size], epochs=epochs, bsize=bsize, p_filepath=preds_filepath, l_filepath=l_filepath, w_filepath=w_filepath, h=h, callbacks=callbacks)

"""
Pure LSTM
"""
if model_choice == "LSTM":
    print("Initializing LSTM object")
    model = noHW_LSTM(inp_window=in_window, out_window=out_window)
    print("Starting LSTM training")
    history, test_predictions, test_real, MSE = model.fit_NN(serie[0:n], serie_test[0:test_size], epochs=epochs, bsize=bsize, p_filepath=preds_filepath, l_filepath=l_filepath, w_filepath=w_filepath, h=h, callbacks=callbacks)

show_result(test_real, test_predictions, full=False)

