import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from pyts.image import GramianAngularField
from pyts.image import MarkovTransitionField

def sequence_splitter(signal, inp_window, out_window, h):
    '''
    Split a signal into a X-Window and a Y-Window
    Returns:
        inp: input_window of shape(#samples,inp_window)
        out: output window of shape(#samples,out_window)
        h: prepare for prediction of y_t+h
    The #samples is a function of the length of the signal, inp_window and out_window parameters.
    '''
    # Prepare windows to fill
    inp = list()
    out = list()

    print("Splitting series into windows")
    for i in tqdm(range(len(signal)- h - out_window)):
        pointer_input = i + inp_window
        pointer_output = pointer_input + h

        if pointer_output+out_window > len(signal):
            break

        window_X = signal[i:pointer_input]
        window_Y = signal[pointer_output:pointer_output+out_window]

        inp.append(window_X)
        out.append(window_Y)

    return np.array(inp), np.array(out)


def show_result(y, y_pred, save_plot=False, full=False, title=""):
    plt.plot(y[150:750], c='red', lw=0.5)
    plt.plot(y_pred[150:750], c='blue')

    if full == True:
        plt.plot(y, c='red')
        plt.plot(y_pred, c='blue')

    plt.title(title)

    if save_plot:
        plt.savefig("plot.png")
    else:
        plt.show()


def prepConvLSTM(seq_len, out_window, img_size, test_size, channels, HW):
    print("Preparing data: ")
    y_test = HW.test_output
    # print(y_test)

    gadf_transformed_test = np.expand_dims(HW.gadf_test, axis=3)
    gasf_transformed_test = np.expand_dims(HW.gasf_test, axis=3)
    mtf_transformed_test = np.expand_dims(HW.mtf_test, axis=3)

    if (channels == 2):
        X_test_windowed = np.concatenate((gadf_transformed_test, gasf_transformed_test), axis=3)

    else:
        X_test_windowed = np.concatenate((gadf_transformed_test, gasf_transformed_test, mtf_transformed_test), axis=3)

    X_test_Conv_LSTM = np.zeros((test_size, seq_len, img_size, img_size, channels))
    y_test_Conv_LSTM = np.zeros((test_size, out_window))

    print("Test data:")
    for i in tqdm(range(0, test_size)):
        current_seq_X = np.zeros((seq_len, img_size, img_size, channels))
        for l in range(seq_len):
            current_seq_X[l] = X_test_windowed[i + l]
        current_seq_X = current_seq_X.reshape(1, seq_len, img_size, img_size, channels)
        X_test_Conv_LSTM[i] = current_seq_X
        y_test_Conv_LSTM[i] = y_test[i + seq_len - 1]

    X_test_Conv_LSTM = X_test_Conv_LSTM.reshape(-1, seq_len, img_size, img_size, channels)
    y_test_Conv_LSTM = y_test_Conv_LSTM.reshape(-1, out_window)

    return (X_test_Conv_LSTM, y_test_Conv_LSTM)

def prep_seriesConvLSTM(seq_len, out_window, in_window, img_size, channels, series_test, h):
    print("Preparing data: ")
    sample_range=(-1,1)
    signal_test = series_test

    signal_test = signal_test.reshape(-1, 1)

    signal_test_scaled = signal_test.flatten()
    window_input_test, window_output_test = sequence_splitter(signal_test_scaled, in_window, out_window, h)

    gadf = GramianAngularField(image_size=img_size, method='difference', sample_range=sample_range)
    gasf = GramianAngularField(image_size=img_size, method='summation', sample_range=sample_range)
    mtf = MarkovTransitionField(image_size=img_size, n_bins=8, strategy='quantile')

    gadf_test = np.expand_dims(gadf.fit_transform(window_input_test), axis=3)
    gasf_test = np.expand_dims(gasf.fit_transform(window_input_test), axis=3)
    mtf_test = np.expand_dims(mtf.fit_transform(window_input_test), axis=3)

    y_test = window_output_test.reshape(-1)

    if(channels==2):
        X_test_windowed = np.concatenate((gadf_test, gasf_test),axis=3)

    else:
        X_test_windowed = np.concatenate((gadf_test, gasf_test, mtf_test), axis=3)

    X_test_Conv_LSTM = np.zeros((X_test_windowed.shape[0] - seq_len+1, seq_len, img_size, img_size, channels))
    y_test_Conv_LSTM = np.zeros((X_test_windowed.shape[0] - seq_len+1, out_window))

    print("Test data:")
    for i in tqdm(range(0, X_test_windowed.shape[0] - seq_len+1)):
        current_seq_X = np.zeros((seq_len, img_size, img_size, channels))
        for l in range(seq_len):
            current_seq_X[l] = X_test_windowed[i + l]
        current_seq_X = current_seq_X.reshape(1, seq_len, img_size, img_size, channels)
        X_test_Conv_LSTM[i] = current_seq_X
        y_test_Conv_LSTM[i] = y_test[i + seq_len - 1]

    X_test_Conv_LSTM = X_test_Conv_LSTM.reshape(-1, seq_len, img_size, img_size, channels)
    y_test_Conv_LSTM = y_test_Conv_LSTM.reshape(-1, out_window)

    return(X_test_Conv_LSTM, y_test_Conv_LSTM)

def prepConvMLP(HW, N_Channel, test_size):
    gadf_transformed_test = np.expand_dims(HW.gadf_test, axis=3)
    gasf_transformed_test = np.expand_dims(HW.gasf_test, axis=3)
    if N_Channel == 3:
        mtf_transformed_test = np.expand_dims(HW.mtf_test, axis=3)

    if N_Channel == 3:
        X_test = np.concatenate((gadf_transformed_test, gasf_transformed_test, mtf_transformed_test), axis=3)[
                 0:test_size]
    else:
        X_test = np.concatenate((gadf_transformed_test, gasf_transformed_test), axis=3)[0:test_size]
    y_test = HW.test_output[0:test_size]
    y_test_raw = HW.test_output_val[0:test_size]

    return(X_test, y_test)

def prep_seriesConvMLP(window_size_x, window_size_y, img_size, signal_test, h):
    signal_test = signal_test.reshape(-1, 1)
    sample_range = (-1, 1)

    signal_test_scaled = signal_test.flatten()

    # Split Sequence
    window_input_test, window_output_test = sequence_splitter(signal_test_scaled, window_size_x, window_size_y, h)

    # %%---------------------------------------------------------------------------
    '''
    Field transformations
    '''

    gadf = GramianAngularField(image_size=img_size, method='difference', sample_range=sample_range)
    gasf = GramianAngularField(image_size=img_size, method='summation', sample_range=sample_range)
    mtf = MarkovTransitionField(image_size=img_size, n_bins=8, strategy='quantile')

    gadf_transformed_test = np.expand_dims(gadf.fit_transform(window_input_test), axis=3)
    gasf_transformed_test = np.expand_dims(gasf.fit_transform(window_input_test), axis=3)
    mtf_transformed_test = np.expand_dims(mtf.fit_transform(window_input_test), axis=3)

    X_test_windowed = np.concatenate((gadf_transformed_test, gasf_transformed_test, mtf_transformed_test), axis=3)

    # Data reshaping

    X_test_Conv_MLP = X_test_windowed
    y_test_Conv_MLP = window_output_test

    return (X_test_Conv_MLP, y_test_Conv_MLP)

def prepLSTM(test, inp_window, out_window, h):
    from sklearn.preprocessing import MinMaxScaler
    MMscaler = MinMaxScaler(feature_range=(-1, 1))
    serie_test = MMscaler.fit_transform(test.reshape(-1, 1)).flatten()

    window_input_test, window_output_test = sequence_splitter(serie_test.reshape(-1, 1), inp_window,
                                                              out_window, h)

    X_test = window_input_test
    y_test = window_output_test

    N_samples_test = np.shape(X_test)[0]

    # Reshape Data

    X_test = X_test.reshape(N_samples_test, inp_window, 1)
    y_test = y_test.reshape(N_samples_test, out_window, 1)

    return(X_test, y_test, MMscaler)