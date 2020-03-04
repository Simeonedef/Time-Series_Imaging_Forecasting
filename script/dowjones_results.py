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

print("======================Conv_LSTM with HW====================")
data = pd.read_csv(os.path.join("predictions_dj", "Conv-LSTM_20200109-1113.csv"))
preds = data['Predictions']
real = data['True value']
MSE_test = ((real - preds) ** 2).mean()
print("MSE Test loss: ", MSE_test)
show_result(real, preds, full=False, title="Conv_LSTM with HW")


print("======================Conv_LSTM without HW====================")
data = pd.read_csv(os.path.join("predictions_dj", "Conv-LSTM_raw_20200109-1111.csv"))
preds = data['Predictions']
real = data['True value']
MSE_test = ((real - preds) ** 2).mean()
print("MSE Test loss: ", MSE_test)
show_result(real, preds, full=False, title="Conv_LSTM without HW")

print("======================Conv_MLP with HW====================")
data = pd.read_csv(os.path.join("predictions_dj", "Conv_MLP-20200109-1128.csv"))
preds = data['Predictions']
real = data['True value']
MSE_test = ((real - preds) ** 2).mean()
print("MSE Test loss: ", MSE_test)
show_result(real, preds, full=False, title="Conv_MLP with HW")

print("======================Conv_MLP without HW====================")
data = pd.read_csv(os.path.join("predictions_dj", "Conv_MLP_raw-20200109-1128.csv"))
preds = data['Predictions']
real = data['True value']
MSE_test = ((real - preds) ** 2).mean()
print("MSE Test loss: ", MSE_test)
show_result(real, preds, full=False, title="Conv_MLP without HW")


print("======================LSTM====================")
data = pd.read_csv(os.path.join("predictions_dj", "LSTM-20200109-1131.csv"))
preds = data['Predictions']
real = data['True value']
MSE_test = ((real - preds) ** 2).mean()
print("MSE Test loss: ", MSE_test)
show_result(real, preds, full=False, title="Pure LSTM")

