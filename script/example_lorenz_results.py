import pandas as pd
import os
from utils import show_result

"""
Retrieving and plotting all results on Stefano Noisy dataset in the following order:
- Conv_LSTM with HW
- Conv_LSTM without HW
- Conv_MLP with HW
- Conv_MLP without HW
- pure LSTM
"""

p_filepath = "predictions_lorenz"
w_filepath = "weights_lorenz"

print("======================Conv_LSTM with HW====================")
data = pd.read_csv(os.path.join(p_filepath, "Conv-LSTM_20200109-0009.csv"))
preds = data['Predictions']
real = data['True value']
MSE_test = ((real - preds) ** 2).mean()
print("MSE Test loss: ", MSE_test)
show_result(real, preds, full=False, title="Conv_LSTM with HW")
#Model path:
#model = load_model(os.path.join(w_filepath,"Conv_LSTM-weights-improvement-10-0.0016-bigger.hdf5"))


print("======================Conv_LSTM without HW====================")
data = pd.read_csv(os.path.join(p_filepath, "Conv-LSTM_raw_20200106-2051.csv"))
preds = data['Predictions']
real = data['True value']
MSE_test = ((real - preds) ** 2).mean()
print("MSE Test loss: ", MSE_test)
show_result(real, preds, full=False, title="Conv_LSTM without HW")
#Model path:
#model = load_model(os.path.join(w_filepath,"Conv_LSTM_raw_weights-18.4356.hdf5"))

print("======================Conv_MLP with HW====================")
data = pd.read_csv(os.path.join(p_filepath, "Conv_MLP-20200108-2118.csv"))
preds = data['Predictions']
real = data['True value']
MSE_test = ((real - preds) ** 2).mean()
print("MSE Test loss: ", MSE_test)
show_result(real, preds, full=False, title="Conv_MLP with HW")
#Model path:
#model = load_model(os.path.join(w_filepath,"Conv_MLP-weights-improvement-10-0.0011-bigger.hdf5"))

print("======================Conv_MLP without HW====================")
data = pd.read_csv(os.path.join(p_filepath, "Conv_MLP_raw-20200108-0841-new.csv"))
preds = data['Predictions']
real = data['True value']
MSE_test = ((real - preds) ** 2).mean()
print("MSE Test loss: ", MSE_test)
show_result(real, preds, full=False, title="Conv_MLP without HW")
#Model path:
#model = load_model(os.path.join(w_filepath,"Conv_MLP_raw-weights-improvement-10-42.5211-bigger-new.hdf5"))

print("======================LSTM====================")
data = pd.read_csv(os.path.join(p_filepath, "LSTM-h10.csv"))
preds = data['Predictions']
real = data['True value']
MSE_test = ((real - preds) ** 2).mean()
print("MSE Test loss: ", MSE_test)
show_result(real, preds, full=False, title="Pure LSTM")
#Model path:
#model = load_model(os.path.join(w_filepath,"LSTM_best_h10.hdf5"))
