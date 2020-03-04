import os
from utils import *
from TSClass_lorenz import *
from noHW_LSTM import *
from holt_winter import *
from tensorflow.keras.models import load_model
from TSClass_function import *
from datetime import datetime
import pandas as pd
from tqdm import tqdm
import argparse
"""
Retrieving and generating results on Noisy dataset in the following order:
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


p_filepath = "predictions_noisy"
w_filepath = "weights_noisy"

n = 6000
dt = 0.1
img_size = 40
seq_len = 3
out_window = 1
test_size = 4000
epochs = 10
bsize = 32
h = 12
in_window = 40
if save:
    save_file = True
else:
    save_file = False


def moving_average(start_idx, end_idx, dt, sigma):
    alpha = 0.35
    beta = 0.15
    gamma = 0.10
    delta = 0.35
    eta = -0.03
    omega = 0.05

    ma = np.zeros(end_idx - start_idx)
    ma[0:6] = np.random.normal(loc=0.0, scale=sigma, size=6)
    for i in range(4, end_idx - start_idx):
        ma[i] = alpha * ma[i - 1] + beta * ma[i - 2] + gamma * ma[i - 3] + delta * ma[i - 4] + eta * ma[i - 5] + \
                omega * ma[i - 6] + np.random.normal(loc=0.0, scale=sigma, size=1)
    return ma


# %%
def function_example_ma(start_idx, end_idx, dt, sigma):
    """
    Example of callable object written the "correct way",
    that accepts as first arguments range of indexes [start_idx, end_idx)
    and return an array of (end_idx-start_idx,) values.
    In this case it is sin + noise
    """
    x1 = start_idx * dt
    x2 = (end_idx - 1) * dt
    m = 28 * dt
    x = np.linspace(x1, x2, end_idx - start_idx)
    return 700 + (160 * np.sin(2 * np.pi * x / (67.345 * m)) + 30 * np.cos(2 * np.pi * x / (3.4 * m))) * \
           (1 + 0.3 * np.sin(2 * np.pi * x / m)) + moving_average(start_idx, end_idx, dt, sigma)

np.random.seed(57)
tsc1 = TSC_function(savevals=True)
tsc2 = TSC_function(savevals=True)

serie  = tsc1.get_next_N_vals(n,function_example_ma, dt, sigma=2.8)
serie_test= tsc2.get_next_N_vals(n,function_example_ma, dt, sigma=2.8)

print("Initializing Holt Winter")
HW = Holt_Winters_NN(serie, serie_test, m=1, h=h, windowsize=img_size, stride=1, alpha=0.35, beta=0.15, gamma=0.1)

X_test_Conv_LSTM, y_test_Conv_LSTM = prepConvLSTM(seq_len, out_window, img_size, test_size, 2, HW)

print("======================Conv_LSTM with HW====================")
model = load_model(os.path.join(w_filepath,"Conv_LSTM-weights-improvement-10-0.0003-bigger.hdf5"))
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
model = load_model(os.path.join(w_filepath,"Conv_LSTM_raw_weights-676.4750-bigger.hdf5"))
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
model = load_model(os.path.join(w_filepath,"Conv_MLP-weights-improvement-10-0.0003-bigger.hdf5"))
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
model = load_model(os.path.join(w_filepath,"Conv_MLP_raw-weights-improvement-10-513.8852-bigger-new.hdf5"))
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

X_test_LSTM, y_test_LSTM, scaler = prepLSTM(serie_test, 100, out_window, h)
print("======================Pure LSTM====================")
model = load_model(os.path.join(w_filepath,"LSTM_weights-0.0165-bigger.hdf5"))
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