import matplotlib.pyplot as plt
from holt_winter import *
from TSClass_function import *
from TSClass_lorenz import *
from Conv_LSTM import *
from utils import show_result

#==============CONSTANTS=============
img_size = 32
seq_len = 3
out_window = 1
preds_filepath = "predictions_test" #Name of folder
n=30000
dt=0.1

in_window = 100
test_size = 10000
tsc = TSC_lorenz(savevals=True)
serie = tsc.get_next_N_vals(n)
serie_test = tsc.get_next_N_vals(n)
# ==============CONSTANTS END=============

print("=============TESTING HW============")
#%%create holt winter object
print("Initializing Holt Winter")
HW=Holt_Winters_NN(serie,serie_test,1,h=6,windowsize=img_size,stride=1,alpha=0.25,beta=0,gamma=0.35)

print("Initializing Conv_LSTM object")
Conv_LSTM_HW = Conv_LSTM(img_size, seq_len, in_window=in_window, out_window=out_window, conv_layers=0, lstm_layers=2, dropout=0.4, pre_loaded=False, bidirectional=True, channels=2, test_size=test_size)
print("Starting Conv_LSTM training")
model_hw, y_true, y_pred = Conv_LSTM_HW.train_HW(HW, epochs=5, bsize=64, p_filepath=preds_filepath, callbacks=False)
print("Finished training")
print("Showing results")
show_result(y_pred, y_true, title="Conv-LSTM With HW", full=False)

print("=============TESTING RAW TS============")
#%%
#create NN model
print("Initializing Conv_LSTM object")
Conv_LSTM_HW = Conv_LSTM(img_size, seq_len, in_window=in_window, out_window=out_window, conv_layers=0, lstm_layers=2, dropout=0.4, pre_loaded=False, bidirectional=True, channels=3, test_size=test_size)
print("Starting Conv_LSTM training")
model_hw, y_true, y_pred = Conv_LSTM_HW.train_series(serie,serie_test, epochs=5, bsize=64, p_filepath=preds_filepath, callbacks=False)
print("Finished training")
print("Showing results")
show_result(y_pred, y_true, title="Conv-LSTM Without HW")
