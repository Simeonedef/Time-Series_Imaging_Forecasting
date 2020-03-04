from utils import show_result
import pandas as pd
import os

"""
Retrieving and plotting all results on Stefano Noisy dataset in the following order:
- Conv_LSTM with HW
- Conv_LSTM without HW
- Conv_MLP with HW
- Conv_MLP without HW
- pure LSTM
"""

p_filepath = "predictions_noisy"
w_filepath = "weights_noisy"

print("======================Conv_LSTM with HW====================")
data = pd.read_csv(os.path.join(p_filepath, "Conv-LSTM_20200108-2319.csv"))
preds = data['Predictions']
real = data['True value']
MSE_test = ((real - preds) ** 2).mean()
print("MSE Test loss: ", MSE_test)
show_result(real, preds, full=False, title="Conv_LSTM with HW")
#Model path:
#model = load_model(os.path.join(w_filepath,"Conv_LSTM-weights-0.0003-bigger.hdf5"))


print("======================Conv_LSTM without HW====================")
data = pd.read_csv(os.path.join(p_filepath, "Conv-LSTM_raw_20200106-1926.csv"))
preds = data['Predictions']
real = data['True value']
MSE_test = ((real - preds) ** 2).mean()
print("MSE Test loss: ", MSE_test)
show_result(real, preds, full=False, title="Conv_LSTM without HW")
#Model path:
#model = load_model(os.path.join(w_filepath,"Conv_LSTM_raw_weights-676.4750-bigger.hdf5"))

print("======================Conv_MLP with HW====================")
data = pd.read_csv(os.path.join(p_filepath, "Conv_MLP-20200108-2133.csv"))
preds = data['Predictions']
real = data['True value']
MSE_test = ((real - preds) ** 2).mean()
print("MSE Test loss: ", MSE_test)
show_result(real, preds, full=False, title="Conv_MLP with HW")
#Model path:
#model = load_model(os.path.join(w_filepath,"Conv_MLP-weights-improvement-10-0.0003-bigger.hdf5"))

print("======================Conv_MLP without HW====================")
data = pd.read_csv(os.path.join(p_filepath, "Conv_MLP_raw-20200108-0829-new.csv"))
preds = data['Predictions']
real = data['True value']
MSE_test = ((real - preds) ** 2).mean()
print("MSE Test loss: ", MSE_test)
show_result(real, preds, full=False, title="Conv_MLP without HW")
#Model path:
#model = load_model(os.path.join(w_filepath,"Conv_MLP_raw-weights-improvement-10-513.8852-bigger-new.hdf5"))

print("======================LSTM====================")
data = pd.read_csv(os.path.join(p_filepath, "LSTM-20200107-1202.csv"))
preds = data['Predictions']
real = data['True value']
MSE_test = ((real - preds) ** 2).mean()
print("MSE Test loss: ", MSE_test)
show_result(real, preds, full=False, title="Pure LSTM")
#Model path:
#model = load_model(os.path.join(w_filepath,"LSTM_weights-0.0165-bigger.hdf5"))
