import os
from utils import *
from TSClass_lorenz import *
from noHW_LSTM import *
from holt_winter import *
from tensorflow.keras.models import load_model
from datetime import datetime
import pandas as pd
from tqdm import tqdm
import argparse
"""
Retrieving and generating results on Lorenz dataset in the following order:
- Conv_LSTM with HW
- Conv_LSTM without HW
- Conv_MLP with HW
- Conv_MLP without HW
- pure LSTM
"""

parser = argparse.ArgumentParser(description="model")
parser.add_argument("--save", help="set it to save predictions to csv", action="store_true")
args = parser.parse_args()
save = args.save


p_filepath = "predictions_lorenz"
w_filepath = "weights_lorenz"

n = 6000
dt = 0.1
img_size = 40
seq_len = 3
out_window = 1
test_size = 4000
epochs = 10
bsize = 32
h = 10
in_window = 40
if save:
    save_file = True
else:
    save_file = False

np.random.seed(57)
tsc = TSC_lorenz(savevals=True)
serie  = 100+tsc.get_next_N_vals(n)
serie_test=100+tsc.get_next_N_vals(n)


print("Initializing Holt Winter")
HW = Holt_Winters_NN(serie, serie_test, m=1, h=h, windowsize=img_size, stride=1, alpha=0.35, beta=0.15, gamma=0.1)

X_test_Conv_LSTM, y_test_Conv_LSTM = prepConvLSTM(seq_len, out_window, img_size, test_size, 2, HW)

print("======================Conv_LSTM with HW====================")
model = load_model(os.path.join(w_filepath,"Conv_LSTM-weights-improvement-10-0.0016-bigger.hdf5"))
preds = model.predict(X_test_Conv_LSTM)
preds = preds.reshape(-1)
forecast_multiplier = HW.forecast_multiplier[seq_len - 1:seq_len - 1 + test_size]
test_predictions = np.multiply(preds, forecast_multiplier)
real = y_test_Conv_LSTM
MSE_test = ((real - test_predictions) ** 2).mean()
print("MSE Test loss: ", MSE_test)
if save_file:
    df = pd.DataFrame({'True value': real.flatten(), 'Predictions': test_predictions.flatten()})
    fpath_p = os.path.join(p_filepath,
                           "preds_gen_Conv-LSTM-HW_" + datetime.now().strftime("%Y%m%d-%H%M") + ".csv")
    df.to_csv(fpath_p)
show_result(real.flatten(), test_predictions.flatten(), full=False, title="Conv_LSTM with HW")

X_test_Conv_LSTM, y_test_Conv_LSTM = prep_seriesConvLSTM(seq_len, out_window, in_window, img_size, 3, serie_test, h)
print("======================Conv_LSTM without HW====================")
model = load_model(os.path.join(w_filepath,"Conv_LSTM_raw_weights-18.4356.hdf5"))
preds = model.predict(X_test_Conv_LSTM)
real = y_test_Conv_LSTM
MSE_test = ((real - preds) ** 2).mean()
print("MSE Test loss: ", MSE_test)
if save_file:
    df = pd.DataFrame({'True value': real.flatten(), 'Predictions': preds.flatten()})
    fpath_p = os.path.join(p_filepath,
                           "preds_gen_Conv-LSTM_" + datetime.now().strftime("%Y%m%d-%H%M") + ".csv")
    df.to_csv(fpath_p)
show_result(real, preds, full=False, title="Conv_LSTM without HW")

X_test_Conv_MLP, y_test_Conv_MLP = prepConvMLP(HW, 2, test_size)
print("======================Conv_MLP with HW====================")
model = load_model(os.path.join(w_filepath,"Conv_MLP-weights-improvement-10-0.0011-bigger.hdf5"))
preds = model.predict(X_test_Conv_MLP)
preds = preds.reshape(-1)
forecast_multiplier = HW.forecast_multiplier[0:test_size]
test_predictions = np.multiply(preds, forecast_multiplier)
real = y_test_Conv_MLP
MSE_test = ((real - test_predictions) ** 2).mean()
print("MSE Test loss: ", MSE_test)
if save_file:
    df = pd.DataFrame({'True value': real.flatten(), 'Predictions': test_predictions.flatten()})
    fpath_p = os.path.join(p_filepath,
                           "preds_gen_Conv-MLP-HW_" + datetime.now().strftime("%Y%m%d-%H%M") + ".csv")
    df.to_csv(fpath_p)
show_result(real.flatten(), test_predictions.flatten(), full=False, title="Conv_MLP with HW")

X_test_Conv_MLP, y_test_Conv_MLP = prep_seriesConvMLP(in_window, out_window, img_size, serie_test, h)
print("======================Conv_MLP without HW====================")
model = load_model(os.path.join(w_filepath,"Conv_MLP_raw-weights-improvement-10-42.5211-bigger-new.hdf5"))
preds = model.predict(X_test_Conv_MLP)
real = y_test_Conv_MLP
MSE_test = ((real - preds) ** 2).mean()
print("MSE Test loss: ", MSE_test)
if save_file:
    df = pd.DataFrame({'True value': real.flatten(), 'Predictions': preds.flatten()})
    fpath_p = os.path.join(p_filepath,
                           "preds_gen_Conv-MLP_" + datetime.now().strftime("%Y%m%d-%H%M") + ".csv")
    df.to_csv(fpath_p)
show_result(real, preds, full=False, title="Conv_MLP without HW")

X_test_LSTM, y_test_LSTM, scaler = prepLSTM(serie_test, in_window, out_window, h)
print("======================Pure LSTM====================")
model = load_model(os.path.join(w_filepath,"LSTM_best_h10.hdf5"))
preds = model.predict(X_test_LSTM)
real = y_test_LSTM
preds = preds[0:test_size].reshape(-1, 1)
real = real[0:test_size].reshape(-1, 1)
preds = scaler.inverse_transform(preds)
real = scaler.inverse_transform(real)
MSE_test = ((real - preds) ** 2).mean()
print("MSE Test loss: ", MSE_test)
if save_file:
    df = pd.DataFrame({'True value': real.flatten(), 'Predictions': preds.flatten()})
    fpath_p = os.path.join("p_filepath",
                           "preds_gen_LSTM_" + datetime.now().strftime("%Y%m%d-%H%M") + ".csv")
    df.to_csv(fpath_p)
show_result(real, preds, full=False, title="Pure LSTM")