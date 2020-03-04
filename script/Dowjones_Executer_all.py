from noHW_LSTM import *
import argparse
from utils import show_result

from holt_winter import *

#%% 
'''
Load Data
'''
timeserie = np.load('data/dowjones_86_19.npy')

serie_train = timeserie[0:4284]
serie_test = timeserie[4284:8568]

#%% Conv_Dense
from Conv_MLP import *
#%% Conv_LSTM
from Conv_LSTM import *


print("###############################")
print("######## Dowjones Data ########")
print("#### Training and testing #####")
print("###############################")

#==============CONSTANTS=============
seq_len = 3
out_window = 1
img_size=30
preds_filepath="predictions_dj"
w_filepath = "weights_dj"
l_filepath ="logs_dj"
in_window = 100
test_size = 4000
epochs = 10
bsize = 32
h=6
#==============CONSTANTS END=============


"""
Initializing Holt-Winter
"""

HW=Holt_Winters_NN(serie_train,serie_test,m=1,h=h,windowsize=img_size,stride=1,alpha=0.25,beta=0,gamma=0.35,pr=1,compute_mtf=True)

# """
# Conv LSTM without HW
# """
# print("Initializing Conv_LSTM object")
# Conv_LSTM_HW = Conv_LSTM(img_size=img_size, seq_length=seq_len, in_window=in_window, out_window=out_window, conv_layers=0, lstm_layers=2, dropout=0.4, pre_loaded=False, bidirectional=True, channels=3, test_size=test_size)
# print("Starting Conv_LSTM training")
# model_hw, test_real, test_predictions = Conv_LSTM_HW.train_series(serie_train,serie_test, epochs=epochs, bsize=bsize, p_filepath=preds_filepath, l_filepath=l_filepath, w_filepath=w_filepath, h=h)
# print("=============Ending Conv_LSTM without HW============")
# show_result(test_real, test_predictions)
#
# """
# Conv LSTM with HW
# """
# print("Initializing Conv_LSTM object")
# Conv_LSTM_HW = Conv_LSTM(img_size=img_size, seq_length=seq_len, in_window=in_window, out_window=1, conv_layers=0, lstm_layers=2, dropout=0.4,
#                          pre_loaded=False, bidirectional=True, channels=2, test_size=test_size)
# print("Starting Conv_LSTM training")
# model_hw, test_real, test_predictions = Conv_LSTM_HW.train_HW(HW, epochs=epochs, bsize=bsize, p_filepath=preds_filepath, l_filepath=l_filepath, w_filepath=w_filepath)
# print("=============Ending Conv_LSTM with HW============")
# show_result(test_real, test_predictions)

"""
Conv MLP with HW
"""
print("Initializing Conv_MLP object")
model = Conv_MLP(img_size=img_size,N_Channel=2, test_size=test_size) #if compute_mtf=False -> set N_Channel=2
print("Starting Conv_MLP training")
history,test_predictions,test_real,MSE = model.train_HW(HW, epochs=epochs, bsize=bsize, p_filepath=preds_filepath, l_filepath=l_filepath, w_filepath=w_filepath)
print("=============Ending Conv_MLP with HW============")
show_result(test_real, test_predictions)
"""
Conv MLP without HW
"""
print("Initializing Conv_MLP object")
model = Conv_MLP(img_size=img_size,N_Channel=3, raw=True, test_size=test_size) #if compute_mtf=False -> set N_Channel=2
print("Starting Conv_MLP training")
history,test_predictions,test_real,MSE = model.train_series(serie_train,serie_test, epochs=epochs, bsize=bsize,  p_filepath=preds_filepath, l_filepath=l_filepath, w_filepath=w_filepath, h=h)
print("=============Ending Conv_MLP without HW============")
show_result(test_real, test_predictions)

"""
Pure LSTM
"""
print("Initializing LSTM object")
model = noHW_LSTM(inp_window=in_window,out_window=out_window)
print("Starting LSTM training")
history,test_predictions,test_real,MSE = model.fit_NN(serie_train,serie_test, epochs=epochs, bsize=bsize, p_filepath=preds_filepath, l_filepath=l_filepath, w_filepath=w_filepath, h=h)
show_result(test_real, test_predictions)
