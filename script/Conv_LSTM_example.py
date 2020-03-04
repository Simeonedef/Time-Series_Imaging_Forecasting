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

def show_result(y, y_pred, title='', full=False):
    plt.plot(y[150:750], c='red', lw=0.5)
    plt.plot(y_pred[150:750], c='blue')
    plt.title(title)
    plt.show()

    if full == True:
        plt.plot(y, c='red')
        plt.plot(y_pred, c='blue')
        plt.show()


#==============CONSTANTS=============
img_size = 32
seq_len = 3
out_window = 1
preds_filepath = "predictions" #Name of folder
n=30000
dt=0.1
#==============CONSTANTS END=============

# print("=============TESTING HW============")
# print("Initializing Conv_LSTM object")
# Conv_LSTM_HW = Conv_LSTM(img_size, seq_len, out_window, conv_layers=0, lstm_layers=2, dropout=0.4, pre_loaded=False, bidirectional=True, channels=2, test_size=3000)
#
#
# # tsc1 = TSC_function(savevals=True)
# # tsc2 = TSC_function(savevals=True)
# tsc3 = TSC_lorenz(savevals=True)
#
# serie  = 100+tsc3.get_next_N_vals(n)
# serie_test=100+tsc3.get_next_N_vals(n)

# #%%create holt winter object
# print("Initializing Holt Winter")
# HW=Holt_Winters_NN(serie,serie_test,1,h=6,windowsize=img_size,stride=1,alpha=0.25,beta=0,gamma=0.35)
# # gc.collect()
# #%%
# #create NN model
# print("Starting Conv_LSTM training")
# # model_hw = Conv_LSTM_HW.train_HW_gen(HW, epochs=1, bsize=32)
#
# model_hw, test_real, test_predictions = Conv_LSTM_HW.train_HW(HW, epochs=2, bsize=32, p_filepath=preds_filepath)
#
# print("Showing results")
# show_result(test_predictions, test_real, title="Post HW plot")

print("=============TESTING RAW TS============")
in_window = 100
print("Initializing Conv_LSTM object")
Conv_LSTM_TS = Conv_LSTM(img_size, seq_len, in_window, out_window, conv_layers=0, lstm_layers=1, dropout=0.4, pre_loaded=False, bidirectional=True, channels=3, test_size=20000)

n_points = 100000
tsc = TSC_lorenz(savevals=True)
serie = tsc.get_next_N_vals(n_points)
signal_train = serie[0:60000]
signal_test = serie[60000:n_points]

#%%
#create NN model
print("Starting Conv_LSTM training")
model_ts = Conv_LSTM_TS.train_series(signal_train, signal_test, epochs=5)
# model_ts = Conv_LSTM_TS.train_series_fresh(signal_train, signal_test, epochs=5)
# model.save("CONV_LSTM.hdf5")
print("Finished training")

# y_test, pred, y_test_prep, preds_prep = Conv_LSTM_TS.get_diff(signal_train, signal_test)

# print("Getting predictions")
# y_preds_ts, y_real_ts= Conv_LSTM_TS.get_predictions_series(signal_test)
#
# print("Showing results")
# show_result(y_preds_ts, y_real_ts)
