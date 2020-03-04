import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from pyts.image import GramianAngularField
from pyts.image import MarkovTransitionField
import os

from holt_winter import *
# from TSClass_function import *
# from TSClass_lorenz import *

import tensorflow as tf
gpu = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu[0], True)
from tensorflow import keras
from datetime import datetime
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Dropout, Flatten, LSTM, GlobalAveragePooling2D, BatchNormalization, TimeDistributed, Input, Bidirectional
import gc

def show_result(y, y_pred, save_plot=False, full=False):
    plt.plot(y[150:750], c='red', lw=0.5)
    plt.plot(y_pred[150:750], c='blue')

    if full == True:
        plt.plot(y, c='red')
        plt.plot(y_pred, c='blue')

    if save_plot:
        plt.savefig("plot.png")
    else:
        plt.show()


def sequence_splitter(signal, inp_window, out_window):
    '''
    Split a signal into a X-Window and a Y-Window
    Returns:
        inp: input_window of shape(#samples,inp_window)
        out: output window of shape(#samples,out_window)
    The #samples is a function of the length of the signal, inp_window and out_window parameters.
    '''
    # Prepare windows to fill
    inp = list()
    out = list()

    for i in range(len(signal)):
        pointer_input = i + inp_window
        pointer_output = pointer_input + out_window

        if pointer_output > len(signal):
            break

        window_X = signal[i:pointer_input]
        window_Y = signal[pointer_input:pointer_output]

        inp.append(window_X)
        out.append(window_Y)

    return np.array(inp), np.array(out)


class Conv_LSTM:
    """
    This class will create a Conv_LSTM network as well as do all the training provided the HW object
    Parameters to set:
    img_size - the size of the image supplied by the transformations
    seq_length - the size of the sequence on which to predict for the LSTM
    in_window - the number of values per window, needed only if we're not using HW
    out_window - the number of output values we want i.e. how much values into the future we want
    conv_layers - the actual number of conv layers will be conv_layers+2 (so default value is 0, which means 2 conv layers)
    lstm_layers - the actual number of lstm layers will be lstm_layers+1 (so default value is 1, which means 2 actual layers)
    dropout - this controls the dropout of the LSTM network
    pre_loaded (bool) - if a model has been saved then set this to True and supply model_path
    model_path - if we're using a preexisting model, then we load it in with this path
    bidirectional (bool) - set this to True if we want the LSTM to be bidirectional (default=True)
    """
    def __init__(self, img_size, seq_length, in_window, out_window=1, conv_layers=0, lstm_layers=1, dropout=0.4, pre_loaded=False, model_path ="", bidirectional=True, channels=3, test_size=300, save_plot=True):
        self.conv_layers = conv_layers
        self.lstm_layer = lstm_layers
        self.bidirectional = bidirectional
        self.img_size = img_size
        self.seq_length = seq_length
        self.out_window = out_window
        self.in_window = in_window
        self.dropout = dropout
        self.model = None
        self.channels = channels
        self.test_size = test_size
        self.save_plot = save_plot

        if(pre_loaded):
            self.model = load_model(model_path)
        else:
            self.model = self.get_model_lstm(out_window, img_size, seq_length, conv_layers, lstm_layers, bidirectional, dropout, channels)
            self.model.compile(optimizer="adam", loss="mse", metrics=['mse'])
            self.model.summary()

    def get_model(self):
        return self.model

    # Conv part
    def get_base_model(self, img_size, conv_layers, channels):
        inp = Input(shape=(img_size, img_size, channels))
        x = BatchNormalization()(inp)
        x = Conv2D(32, kernel_size=(4, 4), activation="elu", padding="same")(x)
        x = BatchNormalization()(x)
        x = Dropout(0.1)(x)
        for i in range(conv_layers):
            x = Conv2D(64, kernel_size=(4,4), padding='same', activation='elu')(x)
            x = BatchNormalization()(x)
            x = MaxPooling2D(padding='same', pool_size=(2,2))(x)
            x = Dropout(0.1)(x)

        final = Conv2D(128, kernel_size=(5, 5), strides=5, activation="elu", padding="same")(x)
        final = BatchNormalization()(final)
        final = Dropout(0.1)(final)

        final = Flatten()(final)

        base_model = Model(inputs=inp, outputs=final)
        base_model.compile(optimizer="adam", loss="mse")
        # base_model.summary()
        return base_model

    # LSTM part
    def get_model_lstm(self, window_size_y, img_size, seq_len, conv_layers, lstm_layers, bidirectional, dropout, channels):
        seq_input = Input(shape=(seq_len, img_size, img_size, channels))
        base_model = self.get_base_model(img_size, conv_layers, channels)
        encoded_sequence = TimeDistributed(base_model)(seq_input)

        encoded_sequence = BatchNormalization()(encoded_sequence)

        for i in range(lstm_layers):
            if(bidirectional):
                encoded_sequence = Bidirectional(
                    LSTM(32, activation="tanh", recurrent_activation='sigmoid', dropout=dropout, recurrent_dropout=dropout,
                         return_sequences=True))(encoded_sequence)

            else:
                encoded_sequence = LSTM(32, activation="tanh", recurrent_activation='sigmoid', dropout=dropout,
                                        recurrent_dropout=dropout, return_sequences=True)(encoded_sequence)

            encoded_sequence = BatchNormalization()(encoded_sequence)

        if(bidirectional):
            encoded_sequence = Bidirectional(
                LSTM(32, activation="tanh", recurrent_activation='sigmoid', dropout=dropout, recurrent_dropout=dropout,
                     return_sequences=False))(encoded_sequence)
        else:
            encoded_sequence = LSTM(32, activation="tanh", recurrent_activation='sigmoid', dropout=dropout, recurrent_dropout=dropout,
                     return_sequences=False)(encoded_sequence)

        encoded_sequence = BatchNormalization()(encoded_sequence)

        # Currently predicting the next value based on seq_len windows
        out = Dense(window_size_y, activation='linear')(encoded_sequence)

        model = Model(seq_input, out)

        return model

    def prep_gen(self, gadf, gasf, mtf, gadf_test, gasf_test, mtf_test, y_train, y_test):
        print("Preparing data: ")
        seq_len = self.seq_length
        out_window = self.out_window
        img_size = self.img_size
        y_test = y_test.reshape(-1)
        test_size = self.test_size
        channels = self.channels
        # print(y_test)

        if(channels==2):
            X_train_windowed = np.concatenate((gadf, gasf),axis=3)
            X_test_windowed = np.concatenate((gadf_test, gasf_test),axis=3)

        else:
            X_train_windowed = np.concatenate((gadf, gasf, mtf), axis=3)
            X_test_windowed = np.concatenate((gadf_test, gasf_test, mtf_test), axis=3)

        X_test_Conv_LSTM = np.zeros((test_size, seq_len, img_size, img_size, channels))
        y_test_Conv_LSTM = np.zeros((test_size, out_window))

        print("Test data:")
        for i in tqdm(range(0, test_size)):
            current_seq_X = np.zeros((seq_len, img_size, img_size, channels))
            for l in range(seq_len):
                current_seq_X[l] =  X_test_windowed[i + l]
            current_seq_X = current_seq_X.reshape(1, seq_len, img_size, img_size, channels)
            X_test_Conv_LSTM[i] = current_seq_X
            y_test_Conv_LSTM[i] = y_test[i + seq_len - 1]

        X_test_Conv_LSTM = X_test_Conv_LSTM.reshape(-1, seq_len, img_size, img_size, channels)
        y_test_Conv_LSTM = y_test_Conv_LSTM.reshape(-1, out_window)

        return(X_train_windowed, y_train, X_test_Conv_LSTM, y_test_Conv_LSTM)

    def prep(self, gadf, gasf, mtf, gadf_test, gasf_test, mtf_test, y_train, y_test):
        print("Preparing data: ")
        seq_len = self.seq_length
        out_window = self.out_window
        img_size = self.img_size
        y_test = y_test.reshape(-1)
        test_size = self.test_size
        channels = self.channels
        # print(y_test)

        if(channels==2):
            X_train_windowed = np.concatenate((gadf, gasf),axis=3)
            X_test_windowed = np.concatenate((gadf_test, gasf_test),axis=3)

        else:
            X_train_windowed = np.concatenate((gadf, gasf, mtf), axis=3)
            X_test_windowed = np.concatenate((gadf_test, gasf_test, mtf_test), axis=3)

        X_train_Conv_LSTM = np.zeros((X_train_windowed.shape[0] - seq_len+1, seq_len, img_size, img_size, channels))
        y_train_Conv_LSTM = np.zeros((X_train_windowed.shape[0] - seq_len+1, out_window))
        X_test_Conv_LSTM = np.zeros((test_size - seq_len+1, seq_len, img_size, img_size, channels))
        y_test_Conv_LSTM = np.zeros((test_size - seq_len+1, out_window))

        print("Train data:")
        for i in tqdm(range(0, X_train_windowed.shape[0] - seq_len+1)):
            current_seq_X = np.zeros((seq_len, img_size, img_size, channels))
            for l in range(seq_len):
                current_seq_X[l] = X_train_windowed[i + l]
            current_seq_X = current_seq_X.reshape(1, seq_len, img_size, img_size, channels)
            # print(current_seq_X)
            X_train_Conv_LSTM[i] = current_seq_X
            y_train_Conv_LSTM[i] = y_train[i + seq_len - 1]

        X_train_Conv_LSTM = X_train_Conv_LSTM.reshape(-1, seq_len, img_size, img_size, channels)
        y_train_Conv_LSTM = y_train_Conv_LSTM.reshape(-1, out_window)

        print("Test data:")
        for i in tqdm(range(0, test_size - seq_len+1)):
            current_seq_X = np.zeros((seq_len, img_size, img_size, channels))
            for l in range(seq_len):
                current_seq_X[l] = X_test_windowed[i + l]
            current_seq_X = current_seq_X.reshape(1, seq_len, img_size, img_size, channels)
            X_test_Conv_LSTM[i] = current_seq_X
            y_test_Conv_LSTM[i] = y_test[i + seq_len - 1]

        X_test_Conv_LSTM = X_test_Conv_LSTM.reshape(-1, seq_len, img_size, img_size, channels)
        y_test_Conv_LSTM = y_test_Conv_LSTM.reshape(-1, out_window)

        return(X_train_Conv_LSTM, y_train_Conv_LSTM, X_test_Conv_LSTM, y_test_Conv_LSTM)

    def prep_series(self, gadf, gasf, mtf, gadf_test, gasf_test, mtf_test, y_train, y_test):
        print("Preparing data: ")
        seq_len = self.seq_length
        out_window = self.out_window
        img_size = self.img_size
        y_test = y_test.reshape(-1)
        channels = self.channels
        # print(y_test)

        if(channels==2):
            X_train_windowed = np.concatenate((gadf, gasf),axis=3)
            X_test_windowed = np.concatenate((gadf_test, gasf_test),axis=3)

        else:
            X_train_windowed = np.concatenate((gadf, gasf, mtf), axis=3)
            X_test_windowed = np.concatenate((gadf_test, gasf_test, mtf_test), axis=3)

        X_train_Conv_LSTM = np.zeros((X_train_windowed.shape[0] - seq_len+1, seq_len, img_size, img_size, channels))
        y_train_Conv_LSTM = np.zeros((X_train_windowed.shape[0] - seq_len+1, out_window))
        X_test_Conv_LSTM = np.zeros((X_test_windowed.shape[0] - seq_len+1, seq_len, img_size, img_size, channels))
        y_test_Conv_LSTM = np.zeros((X_test_windowed.shape[0] - seq_len+1, out_window))

        print("Train data:")
        for i in tqdm(range(0, X_train_windowed.shape[0] - seq_len+1)):
            current_seq_X = np.zeros((seq_len, img_size, img_size, channels))
            for l in range(seq_len):
                current_seq_X[l] = X_train_windowed[i + l]
            current_seq_X = current_seq_X.reshape(1, seq_len, img_size, img_size, channels)
            # print(current_seq_X)
            X_train_Conv_LSTM[i] = current_seq_X
            y_train_Conv_LSTM[i] = y_train[i + seq_len - 1]

        X_train_Conv_LSTM = X_train_Conv_LSTM.reshape(-1, seq_len, img_size, img_size, channels)
        y_train_Conv_LSTM = y_train_Conv_LSTM.reshape(-1, out_window)

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

        return(X_train_Conv_LSTM, y_train_Conv_LSTM, X_test_Conv_LSTM, y_test_Conv_LSTM)

    def gen(self, batch_size, X_train, y_train):
        print("In generator")
        seq_len = self.seq_length
        img_size = self.img_size
        channels = self.channels
        out_window = self.out_window

        X_train_Conv_LSTM = np.zeros((batch_size, seq_len, img_size, img_size, channels))
        y_train_Conv_LSTM = np.zeros((batch_size, out_window))

        while 1:
            for i in range((y_train.shape[0]//batch_size) - 1):
                for j in range(0, batch_size):
                    current_seq_X = np.zeros((seq_len, img_size, img_size, channels))
                    for l in range(seq_len):
                        current_seq_X[l] = X_train[i*batch_size + l]
                    current_seq_X = current_seq_X.reshape(1, seq_len, img_size, img_size, channels)
                    # print(current_seq_X)
                    X_train_Conv_LSTM[j] = current_seq_X
                    y_train_Conv_LSTM[j] = y_train[i*batch_size + seq_len - 1]

                X_train_Conv_LSTM = X_train_Conv_LSTM.reshape(batch_size, seq_len, img_size, img_size, channels)
                y_train_Conv_LSTM = y_train_Conv_LSTM.reshape(batch_size, out_window)

                yield X_train_Conv_LSTM, y_train_Conv_LSTM

    def train_HW_gen(self, HW, epochs=5, bsize=8, p_filepath="predictions"):
        Conv_LSTM_model = self.model
        seq_len = self.seq_length
        channels = self.channels
        test_size = self.test_size

        import os
        path = os.path.join("logs", "scalars")

        logdir = path + datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)
        path_w = ("weights")
        filepath = os.path.join(path_w, "weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5")
        checkpoint = keras.callbacks.ModelCheckpoint(
            filepath,
            monitor='loss',
            verbose=0,
            save_best_only=True,
            mode='min'
        )
        callbacks_list = [checkpoint, tensorboard_callback]

        gadf_transformed_train = np.expand_dims(HW.gadf, axis=3)
        gasf_transformed_train = np.expand_dims(HW.gasf, axis=3)
        mtf_transformed_train = np.expand_dims(HW.mtf, axis=3)
        y_train = HW.obtain_training_output()

        gadf_transformed_test = np.expand_dims(HW.gadf_test, axis=3)
        gasf_transformed_test = np.expand_dims(HW.gasf_test, axis=3)
        mtf_transformed_test = np.expand_dims(HW.mtf_test, axis=3)

        y_test = HW.test_output_val
        # print(y_test)

        X_train_Conv_LSTM, y_train_Conv_LSTM, X_test_Conv_LSTM, y_test_Conv_LSTM = self.prep_gen(gadf_transformed_train, gasf_transformed_train, mtf_transformed_train,
                                                                                             gadf_transformed_test, gasf_transformed_test, mtf_transformed_test,
                                                                                             y_train, y_test)

        # print(y_test_Conv_LSTM)
        print("Training has started, launch tensorboard with: %tensorboard --logdir logs/scalar")
        history_Conv_LSTM = Conv_LSTM_model.fit_generator(self.gen(bsize,X_train_Conv_LSTM, y_train_Conv_LSTM), steps_per_epoch=((y_train_Conv_LSTM.shape[0]-seq_len+1)//bsize), epochs=epochs,
                                                callbacks=callbacks_list, validation_data=(X_test_Conv_LSTM, y_test_Conv_LSTM), validation_steps=(X_test_Conv_LSTM.shape[0] // bsize),
                                                shuffle=True, verbose=1)
        print("Average test loss: ", np.average(history_Conv_LSTM.history['loss']))

        preds = Conv_LSTM_model.predict(X_test_Conv_LSTM)
        preds = preds.reshape(-1)
        y_test_Conv_LSTM = y_test_Conv_LSTM.reshape(-1)
        MSE_test_no_HW = ((y_test_Conv_LSTM - preds) ** 2).mean()
        print("Test loss without HW multiplier: ", MSE_test_no_HW)
        # show_result(y_test_Conv_LSTM, preds)

        forecast_multiplier = HW.forecast_multiplier[seq_len - 1:seq_len - 1 + test_size]
        test_predictions = np.multiply(preds, forecast_multiplier)
        test_real = np.multiply(y_test_Conv_LSTM, forecast_multiplier)
        MSE_test = ((test_real - test_predictions) ** 2).mean()

        print("Test loss with HW multiplier: ", MSE_test)
        df = pd.DataFrame({'True value': test_real.flatten(), 'Predictions': test_predictions.flatten()})
        fpath_p = os.path.join(p_filepath, "predictions" + datetime.now().strftime("%Y%m%d-%H%M%S") + ".csv")
        df.to_csv(fpath_p)
        # show_result(test_real, test_predictions)


        return Conv_LSTM_model, test_real, test_predictions

    def train_HW(self, HW, epochs=5, bsize=8, p_filepath="predictions"):
        Conv_LSTM_model = self.model
        seq_len = self.seq_length
        channels = self.channels
        test_size = self.test_size

        path = os.path.join("logs", "scalars_HW_")

        logdir = path + datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

        path_w = ("weights")
        filepath = os.path.join(path_w, "weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5")
        checkpoint = keras.callbacks.ModelCheckpoint(
            filepath,
            monitor='loss',
            verbose=0,
            save_best_only=True,
            mode='min'
        )
        callbacks_list = [checkpoint, tensorboard_callback]

        gadf_transformed_train = np.expand_dims(HW.gadf, axis=3)
        gasf_transformed_train = np.expand_dims(HW.gasf, axis=3)
        mtf_transformed_train = np.expand_dims(HW.mtf, axis=3)
        y_train = HW.obtain_training_output()

        gadf_transformed_test = np.expand_dims(HW.gadf_test, axis=3)
        gasf_transformed_test = np.expand_dims(HW.gasf_test, axis=3)
        mtf_transformed_test = np.expand_dims(HW.mtf_test, axis=3)

        y_test = HW.test_output_val

        X_train_Conv_LSTM, y_train_Conv_LSTM, X_test_Conv_LSTM, y_test_Conv_LSTM = self.prep(gadf_transformed_train, gasf_transformed_train, mtf_transformed_train,
                                                                                             gadf_transformed_test, gasf_transformed_test, mtf_transformed_test,
                                                                                             y_train, y_test)

        # print(y_test_Conv_LSTM)
        print("Training has started, launch tensorboard with: %tensorboard --logdir logs/scalar")
        history_Conv_LSTM = Conv_LSTM_model.fit(x=X_train_Conv_LSTM, y=y_train_Conv_LSTM, batch_size=bsize, epochs=epochs,
                                                callbacks=callbacks_list, validation_data=(X_test_Conv_LSTM, y_test_Conv_LSTM),
                                                shuffle=True, verbose=1)
        print("Average test loss: ", np.average(history_Conv_LSTM.history['loss']))

        preds = Conv_LSTM_model.predict(X_test_Conv_LSTM)
        preds = preds.reshape(-1)
        y_test_Conv_LSTM = y_test_Conv_LSTM.reshape(-1)
        MSE_test_no_HW = ((y_test_Conv_LSTM - preds) ** 2).mean()
        print("Test loss without HW multiplier: ", MSE_test_no_HW)
        # show_result(y_test_Conv_LSTM, preds)

        forecast_multiplier = HW.forecast_multiplier[seq_len - 1:seq_len - 1 + test_size]
        test_predictions = np.multiply(preds, forecast_multiplier)
        test_real = np.multiply(y_test_Conv_LSTM, forecast_multiplier)
        MSE_test = ((test_real - test_predictions) ** 2).mean()

        print("Test loss with HW multiplier: ", MSE_test)
        # show_result(test_real, test_predictions)
        df = pd.DataFrame({'True value': test_real.flatten(), 'Predictions': test_predictions.flatten()})
        fpath_p = os.path.join(p_filepath, "predictions"+datetime.now().strftime("%Y%m%d-%H%M%S")+".csv")
        df.to_csv(fpath_p)

        return Conv_LSTM_model, test_real, test_predictions

    def train_series(self, series_train, series_test, p_filepath="predictions", epochs=5, bsize=16):
        img_size = self.img_size
        Conv_LSTM_model = self.model
        out_window = self.out_window
        in_window = self.in_window
        sample_range = (-1,1)

        path = os.path.join("logs", "scalars_raw_")

        logdir = path + datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

        path_w = ("weights")
        filepath = os.path.join(path_w, "weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5")
        checkpoint = keras.callbacks.ModelCheckpoint(
            filepath,
            monitor='loss',
            verbose=0,
            save_best_only=True,
            mode='min'
        )
        callbacks_list = [checkpoint, tensorboard_callback]

        signal_train = series_train
        signal_test = series_test

        signal_train = signal_train.reshape(-1, 1)
        signal_test = signal_test.reshape(-1, 1)
        from sklearn.preprocessing import MinMaxScaler
        MMscaler = MinMaxScaler(feature_range=sample_range)

        signal_train_scaled = MMscaler.fit_transform(signal_train).flatten()
        signal_test_scaled = MMscaler.fit_transform(signal_test).flatten()
        window_input_train, window_output_train = sequence_splitter(signal_train_scaled, in_window, out_window)
        window_input_test, window_output_test = sequence_splitter(signal_test_scaled, in_window, out_window)

        gadf = GramianAngularField(image_size=img_size, method='difference', sample_range=sample_range)
        gasf = GramianAngularField(image_size=img_size, method='summation', sample_range=sample_range)
        mtf = MarkovTransitionField(image_size=img_size, n_bins=8, strategy='quantile')

        gadf_transformed_train = np.expand_dims(gadf.fit_transform(window_input_train), axis=3)
        gasf_transformed_train = np.expand_dims(gasf.fit_transform(window_input_train), axis=3)
        mtf_transformed_train = np.expand_dims(mtf.fit_transform(window_input_train), axis=3)

        gadf_transformed_test = np.expand_dims(gadf.fit_transform(window_input_test), axis=3)
        gasf_transformed_test = np.expand_dims(gasf.fit_transform(window_input_test), axis=3)
        mtf_transformed_test = np.expand_dims(mtf.fit_transform(window_input_test), axis=3)

        X_train_Conv_LSTM, y_train_Conv_LSTM, X_test_Conv_LSTM, y_test_Conv_LSTM = self.prep_series(gadf_transformed_train,
                                                                                             gasf_transformed_train,
                                                                                             mtf_transformed_train,
                                                                                             gadf_transformed_test,
                                                                                             gasf_transformed_test,
                                                                                             mtf_transformed_test,
                                                                                             window_output_train, window_input_test)

        history_Conv_LSTM = Conv_LSTM_model.fit(x=X_train_Conv_LSTM, y=y_train_Conv_LSTM, batch_size=16,
                                                epochs=5,
                                                callbacks=None, shuffle=True,
                                                validation_data=(X_test_Conv_LSTM, y_test_Conv_LSTM))

        print("Average test loss: ", np.average(history_Conv_LSTM.history['val_loss']))

        preds = Conv_LSTM_model.predict(X_test_Conv_LSTM)
        preds = preds.reshape(-1)
        y_test_Conv_LSTM = y_test_Conv_LSTM.reshape(-1)
        MSE_test_no_HW = ((y_test_Conv_LSTM - preds) ** 2).mean()
        print("Test loss: ", MSE_test_no_HW)
        show_result(y_test_Conv_LSTM, preds)

        df = pd.DataFrame({'True value': y_test_Conv_LSTM.flatten(), 'Predictions': preds.flatten()})
        fpath_p = os.path.join(p_filepath, "predictions_raw_" + datetime.now().strftime("%Y%m%d-%H%M%S") + ".csv")
        df.to_csv(fpath_p)

        return Conv_LSTM_model, y_test_Conv_LSTM, preds


    def train_series_2(self, series_train, series_test, p_filepath="predictions", epochs=5, bsize=16):
        img_size = self.img_size
        Conv_LSTM_model = self.model
        out_window = self.out_window
        in_window = self.in_window
        sample_range = (-1, 1)

        signal_train = series_train
        signal_test = series_test

        signal_train = signal_train.reshape(-1, 1)
        signal_test = signal_test.reshape(-1, 1)
        from sklearn.preprocessing import MinMaxScaler
        MMscaler = MinMaxScaler(feature_range=sample_range)

        signal_train_scaled = MMscaler.fit_transform(signal_train).flatten()
        signal_test_scaled = MMscaler.fit_transform(signal_test).flatten()
        window_input_train, window_output_train = sequence_splitter(signal_train_scaled, in_window, out_window)
        window_input_test, window_output_test = sequence_splitter(signal_test_scaled, in_window, out_window)

        gadf = GramianAngularField(image_size=img_size, method='difference', sample_range=sample_range)
        gasf = GramianAngularField(image_size=img_size, method='summation', sample_range=sample_range)
        mtf = MarkovTransitionField(image_size=img_size, n_bins=8, strategy='quantile')

        gadf_transformed_train = np.expand_dims(gadf.fit_transform(window_input_train), axis=3)
        gasf_transformed_train = np.expand_dims(gasf.fit_transform(window_input_train), axis=3)
        mtf_transformed_train = np.expand_dims(mtf.fit_transform(window_input_train), axis=3)

        gadf_transformed_test = np.expand_dims(gadf.fit_transform(window_input_test), axis=3)
        gasf_transformed_test = np.expand_dims(gasf.fit_transform(window_input_test), axis=3)
        mtf_transformed_test = np.expand_dims(mtf.fit_transform(window_input_test), axis=3)

        X_train_prep, y_train_prep, X_test_prep, y_test_prep = self.prep_series(gadf_transformed_train,
                                                                                gasf_transformed_train,
                                                                                mtf_transformed_train,
                                                                                gadf_transformed_test,
                                                                                gasf_transformed_test,
                                                                                mtf_transformed_test,
                                                                                window_output_train, window_output_test)

        history_Conv_LSTM_prep = Conv_LSTM_model.fit(x=X_train_prep, y=y_train_prep, batch_size=bsize,
                                                     epochs=epochs,
                                                     callbacks=None, shuffle=True,
                                                     validation_data=(X_test_prep, y_test_prep))

        print("Average test loss prep: ", np.average(history_Conv_LSTM_prep.history['val_loss']))

        preds_prep = Conv_LSTM_model.predict(X_test_prep)
        preds_prep = preds_prep.reshape(-1)
        y_test_prep = y_test_prep.reshape(-1)
        MSE_test_no_HW_prep = ((y_test_prep - preds_prep) ** 2).mean()
        print("Test loss prep: ", MSE_test_no_HW_prep)
        show_result(y_test_prep, preds_prep)

        df = pd.DataFrame({'True value': y_test_prep.flatten(), 'Predictions': preds_prep.flatten()})
        fpath_p = os.path.join(p_filepath,
                               "predictions_raw_" + datetime.now().strftime("%Y%m%d-%H%M%S") + ".csv")
        df.to_csv(fpath_p)

        return y_test_prep, preds_prep

    def get_predictions_series(self, series):
        seq_len = self.seq_length
        img_size = self.img_size
        Conv_LSTM_model = self.model
        out_window = self.out_window
        in_window = self.in_window
        sample_range = (-1,1)

        signal_test = series.reshape(-1, 1)
        from sklearn.preprocessing import MinMaxScaler
        MMscaler = MinMaxScaler(feature_range=sample_range)

        signal_test_scaled = MMscaler.fit_transform(signal_test).flatten()
        window_input_test, window_output_test = sequence_splitter(signal_test_scaled, in_window, out_window)

        gadf = GramianAngularField(image_size=img_size, method='difference', sample_range=sample_range)
        gasf = GramianAngularField(image_size=img_size, method='summation', sample_range=sample_range)
        mtf = MarkovTransitionField(image_size=img_size, n_bins=8, strategy='quantile')

        gadf_transformed_train = np.expand_dims(gadf.fit_transform(window_input_test), axis=3)
        gasf_transformed_train = np.expand_dims(gasf.fit_transform(window_input_test), axis=3)
        mtf_transformed_train = np.expand_dims(mtf.fit_transform(window_input_test), axis=3)

        X_test_windowed = np.concatenate((gadf_transformed_train, gasf_transformed_train, mtf_transformed_train),
                                          axis=3)

        y_test = window_output_test
        X_test_Conv_LSTM = []
        y_test_Conv_LSTM = []

        for i in range(0, X_test_windowed.shape[0] - seq_len):
            current_seq_X = []
            for l in range(seq_len):
                current_seq_X.append(X_test_windowed[i + l])
            X_test_Conv_LSTM.append(current_seq_X)
            y_test_Conv_LSTM.append(y_test[i+seq_len-1])

        X_test_Conv_LSTM = np.array(X_test_Conv_LSTM)
        X_test_Conv_LSTM = X_test_Conv_LSTM.reshape(-1, seq_len, img_size, img_size, 3)
        y_test_Conv_LSTM = np.array(y_test_Conv_LSTM)
        y_test_Conv_LSTM = y_test_Conv_LSTM.reshape(-1, out_window)

        test_predictions = Conv_LSTM_model.predict(X_test_Conv_LSTM)
        test_predictions = test_predictions.reshape(-1, 1)
        test_predictions = test_predictions.flatten()
        test_real = y_test_Conv_LSTM
        # MSE_test = ((test_real - test_predictions) ** 2).mean()

        return (test_predictions, test_real)

    def get_predictions_HW(self, HW, val=True):
        seq_len = self.seq_length
        img_size = self.img_size
        model = self.model
        test_size = self.test_size

        if(val):
            X_test_Conv_LSTM = self.for_predict_x
            y_test = self.for_predict_y

            test_predictions_tmp = model.predict(X_test_Conv_LSTM)
            test_predictions_tmp = test_predictions_tmp.reshape(-1)
            # test_predictions_tmp = test_predictions_tmp.flatten()
            forecast_multiplier = HW.forecast_multiplier[seq_len-1:test_size]
            test_predictions = np.multiply(test_predictions_tmp, forecast_multiplier)
            test_real = np.multiply(y_test, forecast_multiplier)
            MSE_test = ((test_real - test_predictions) ** 2).mean()

            return (MSE_test, test_predictions, test_real, test_predictions_tmp, y_test)

        else:
            y_test = HW.test_output_val
            gadf_transformed_test = np.expand_dims(HW.gadf_test, axis=3)
            gasf_transformed_test = np.expand_dims(HW.gasf_test, axis=3)
            mtf_transformed_test = np.expand_dims(HW.mtf_test, axis=3)
            X_test_windowed = np.concatenate((gadf_transformed_test, gasf_transformed_test, mtf_transformed_test),
                                             axis=3)
            X_test_Conv_LSTM = []
            for i in range(0, X_test_windowed.shape[0] - seq_len):
                current_seq_X = []
                for l in range(seq_len):
                    current_seq_X.append(X_test_windowed[i + l])
                X_test_Conv_LSTM.append(current_seq_X)

            X_test_Conv_LSTM = np.array(X_test_Conv_LSTM)
            X_test_Conv_LSTM = X_test_Conv_LSTM.reshape(-1, seq_len, img_size, img_size, 3)

            test_predictions_tmp = model.predict(X_test_Conv_LSTM)
            test_predictions_tmp = test_predictions_tmp.reshape(-1, 1)
            test_predictions_tmp = test_predictions_tmp.flatten()
            forecast_multiplier = HW.forecast_multiplier[seq_len:]
            test_predictions = test_predictions_tmp * forecast_multiplier
            test_real = HW.test_output[seq_len:]
            MSE_test = ((test_real - test_predictions) ** 2).mean()

            return (MSE_test, test_predictions, test_real, test_predictions_tmp, y_test)

    def train_series_fresh(self, series_train, series_test, epochs=5, bsize=32, p_filepath="predictions"):
        img_size = self.img_size
        Conv_LSTM_model = self.model
        out_window = self.out_window
        in_window = self.in_window
        sample_range = (-1, 1)
        seq_len = self.seq_length
        channels = self.channels

        path = os.path.join("logs", "scalars_raw_fresh_")

        logdir = path + datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

        path_w = ("weights")
        filepath = os.path.join(path_w, "weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5")
        checkpoint = keras.callbacks.ModelCheckpoint(
            filepath,
            monitor='loss',
            verbose=0,
            save_best_only=True,
            mode='min'
        )
        callbacks_list = [checkpoint, tensorboard_callback]

        signal_train = series_train
        signal_test = series_test

        signal_train = signal_train.reshape(-1, 1)
        signal_test = signal_test.reshape(-1, 1)
        from sklearn.preprocessing import MinMaxScaler
        MMscaler = MinMaxScaler(feature_range=sample_range)

        signal_train_scaled = MMscaler.fit_transform(signal_train).flatten()
        signal_test_scaled = MMscaler.fit_transform(signal_test).flatten()
        window_input_train, window_output_train = sequence_splitter(signal_train_scaled, in_window, out_window)
        window_input_test, window_output_test = sequence_splitter(signal_test_scaled, in_window, out_window)

        gadf = GramianAngularField(image_size=img_size, method='difference', sample_range=sample_range)
        gasf = GramianAngularField(image_size=img_size, method='summation', sample_range=sample_range)
        mtf = MarkovTransitionField(image_size=img_size, n_bins=8, strategy='quantile')

        gadf_transformed_train = np.expand_dims(gadf.fit_transform(window_input_train), axis=3)
        gasf_transformed_train = np.expand_dims(gasf.fit_transform(window_input_train), axis=3)
        mtf_transformed_train = np.expand_dims(mtf.fit_transform(window_input_train), axis=3)

        gadf_transformed_test = np.expand_dims(gadf.fit_transform(window_input_test), axis=3)
        gasf_transformed_test = np.expand_dims(gasf.fit_transform(window_input_test), axis=3)
        mtf_transformed_test = np.expand_dims(mtf.fit_transform(window_input_test), axis=3)

        X_train_windowed = np.concatenate((gadf_transformed_train, gasf_transformed_train, mtf_transformed_train), axis=3)
        X_test_windowed = np.concatenate((gadf_transformed_test, gasf_transformed_test, mtf_transformed_test), axis=3)

        X_train_Conv_LSTM = []
        y_train_Conv_LSTM = []
        X_test_Conv_LSTM = []
        y_test_Conv_LSTM = []

        for i in range(0, X_train_windowed.shape[0] - seq_len+1):
            current_seq_X = []
            for l in range(seq_len):
                current_seq_X.append(X_train_windowed[i + l])
            X_train_Conv_LSTM.append(current_seq_X)
            y_train_Conv_LSTM.append(window_output_train[i + seq_len - 1])

        X_train_Conv_LSTM = np.array(X_train_Conv_LSTM)
        X_train_Conv_LSTM = X_train_Conv_LSTM.reshape(-1, seq_len, img_size, img_size, channels)
        y_train_Conv_LSTM = np.array(y_train_Conv_LSTM)
        y_train_Conv_LSTM = y_train_Conv_LSTM.reshape(-1, out_window)

        for i in range(0, X_test_windowed.shape[0] - seq_len+1):
            current_seq_X = []
            for l in range(seq_len):
                current_seq_X.append(X_test_windowed[i + l])
            X_test_Conv_LSTM.append(current_seq_X)
            y_test_Conv_LSTM.append(window_output_test[i + seq_len - 1])

        X_test_Conv_LSTM = np.array(X_test_Conv_LSTM)
        X_test_Conv_LSTM = X_test_Conv_LSTM.reshape(-1, seq_len, img_size, img_size, channels)
        y_test_Conv_LSTM = np.array(y_test_Conv_LSTM)
        y_test_Conv_LSTM = y_test_Conv_LSTM.reshape(-1, out_window)

        history_Conv_LSTM = Conv_LSTM_model.fit(x=X_train_Conv_LSTM, y=y_train_Conv_LSTM, batch_size=bsize,
                                                epochs=epochs,
                                                callbacks=callbacks_list, shuffle=True,
                                                validation_data=(X_test_Conv_LSTM, y_test_Conv_LSTM))

        print("Average test loss: ", np.average(history_Conv_LSTM.history['val_loss']))

        preds = Conv_LSTM_model.predict(X_test_Conv_LSTM)
        preds = preds.reshape(-1)
        y_test_Conv_LSTM = y_test_Conv_LSTM.reshape(-1)
        MSE_test_no_HW = ((y_test_Conv_LSTM - preds) ** 2).mean()
        print("Test loss: ", MSE_test_no_HW)
        show_result(y_test_Conv_LSTM, preds)

        df = pd.DataFrame({'True value': y_test_Conv_LSTM.flatten(), 'Predictions': preds.flatten()})
        fpath_p = os.path.join(p_filepath, "predictions_raw_fresh_" + datetime.now().strftime("%Y%m%d-%H%M%S") + ".csv")
        df.to_csv(fpath_p)

        return Conv_LSTM_model, y_test_Conv_LSTM, preds

    def get_diff(self, series_train, series_test):
        img_size = self.img_size
        Conv_LSTM_model = self.model
        out_window = self.out_window
        in_window = self.in_window
        sample_range = (-1, 1)
        seq_len = self.seq_length
        channels = self.channels

        signal_train = series_train
        signal_test = series_test

        signal_train = signal_train.reshape(-1, 1)
        signal_test = signal_test.reshape(-1, 1)
        from sklearn.preprocessing import MinMaxScaler
        MMscaler = MinMaxScaler(feature_range=sample_range)

        signal_train_scaled = MMscaler.fit_transform(signal_train).flatten()
        signal_test_scaled = MMscaler.fit_transform(signal_test).flatten()
        window_input_train, window_output_train = sequence_splitter(signal_train_scaled, in_window, out_window)
        window_input_test, window_output_test = sequence_splitter(signal_test_scaled, in_window, out_window)

        gadf = GramianAngularField(image_size=img_size, method='difference', sample_range=sample_range)
        gasf = GramianAngularField(image_size=img_size, method='summation', sample_range=sample_range)
        mtf = MarkovTransitionField(image_size=img_size, n_bins=8, strategy='quantile')

        gadf_transformed_train = np.expand_dims(gadf.fit_transform(window_input_train), axis=3)
        gasf_transformed_train = np.expand_dims(gasf.fit_transform(window_input_train), axis=3)
        mtf_transformed_train = np.expand_dims(mtf.fit_transform(window_input_train), axis=3)

        gadf_transformed_test = np.expand_dims(gadf.fit_transform(window_input_test), axis=3)
        gasf_transformed_test = np.expand_dims(gasf.fit_transform(window_input_test), axis=3)
        mtf_transformed_test = np.expand_dims(mtf.fit_transform(window_input_test), axis=3)

        X_train_windowed = np.concatenate((gadf_transformed_train, gasf_transformed_train, mtf_transformed_train),
                                          axis=3)
        X_test_windowed = np.concatenate((gadf_transformed_test, gasf_transformed_test, mtf_transformed_test), axis=3)

        X_train_Conv_LSTM = []
        y_train_Conv_LSTM = []
        X_test_Conv_LSTM = []
        y_test_Conv_LSTM = []

        print("Getting Train original")
        for i in tqdm(range(0, X_train_windowed.shape[0] - seq_len + 1)):
            current_seq_X = []
            for l in range(seq_len):
                current_seq_X.append(X_train_windowed[i + l])
            X_train_Conv_LSTM.append(current_seq_X)
            y_train_Conv_LSTM.append(window_output_train[i + seq_len - 1])

        X_train_Conv_LSTM = np.array(X_train_Conv_LSTM)
        X_train_Conv_LSTM = X_train_Conv_LSTM.reshape(-1, seq_len, img_size, img_size, channels)
        y_train_Conv_LSTM = np.array(y_train_Conv_LSTM)
        y_train_Conv_LSTM = y_train_Conv_LSTM.reshape(-1, out_window)

        print("Getting test original")
        for i in tqdm(range(0, X_test_windowed.shape[0] - seq_len + 1)):
            current_seq_X = []
            for l in range(seq_len):
                current_seq_X.append(X_test_windowed[i + l])
            X_test_Conv_LSTM.append(current_seq_X)
            y_test_Conv_LSTM.append(window_output_test[i + seq_len - 1])

        X_test_Conv_LSTM = np.array(X_test_Conv_LSTM)
        X_test_Conv_LSTM = X_test_Conv_LSTM.reshape(-1, seq_len, img_size, img_size, channels)
        y_test_Conv_LSTM = np.array(y_test_Conv_LSTM)
        y_test_Conv_LSTM = y_test_Conv_LSTM.reshape(-1, out_window)

        X_train_prep, y_train_prep, X_test_prep, y_test_prep = self.prep_series(gadf_transformed_train,
                                                                         gasf_transformed_train,
                                                                         mtf_transformed_train,
                                                                         gadf_transformed_test,
                                                                         gasf_transformed_test,
                                                                         mtf_transformed_test,
                                                                         window_output_train, window_output_test)

        # df = pd.DataFrame({'x_test_prep_c3': X_train_prep[:,0,15,15,2].flatten(), 'x_test_c3': X_train_Conv_LSTM[:,0,15,15,2].flatten(), 'x_test_prep_c2': X_train_prep[:,0,15,15,1].flatten(), 'x_test_c2': X_train_Conv_LSTM[:,0,15,15,1].flatten()})
        # df.to_csv("test.csv")

        history_Conv_LSTM_prep = Conv_LSTM_model.fit(x=X_train_prep, y=y_train_prep, batch_size=16,
                                                epochs=5,
                                                callbacks=None, shuffle=True,
                                                validation_data=(X_test_prep, y_test_prep))

        print("Average test loss prep: ", np.average(history_Conv_LSTM_prep.history['val_loss']))

        preds_prep = Conv_LSTM_model.predict(X_train_prep)
        preds_prep = preds_prep.reshape(-1)
        y_test_prep = y_test_prep.reshape(-1)
        MSE_test_no_HW_prep = ((y_test_prep - preds_prep) ** 2).mean()
        print("Test loss prep: ", MSE_test_no_HW_prep)
        show_result(y_test_prep, preds_prep)

        df = pd.DataFrame({'True value': y_test_prep.flatten(), 'Predictions': preds_prep.flatten()})
        fpath_p = os.path.join("predictions",
                               "predictions_raw_prep_" + datetime.now().strftime("%Y%m%d-%H%M%S") + ".csv")
        df.to_csv(fpath_p)


        model2 = self.get_model_lstm(out_window, img_size, 3, 0, 1, bidirectional=True,
                                         dropout=0.4, channels=3)
        model2.compile(optimizer="adam", loss="mse", metrics=['mse'])
        history_Conv_LSTM = model2.fit(x=X_train_Conv_LSTM, y=y_train_Conv_LSTM, batch_size=16,
                                            epochs=5,callbacks=None, shuffle=True,
                                            validation_data=(X_test_Conv_LSTM, y_test_Conv_LSTM))

        print("Average test loss: ", np.average(history_Conv_LSTM.history['val_loss']))

        preds = model2.predict(X_test_Conv_LSTM)
        preds = preds.reshape(-1)
        y_test_Conv_LSTM = y_test_Conv_LSTM.reshape(-1)
        MSE_test_no_HW = ((y_test_Conv_LSTM - preds) ** 2).mean()
        print("Test loss: ", MSE_test_no_HW)
        show_result(y_test_Conv_LSTM, preds)

        df = pd.DataFrame({'True value': y_test_Conv_LSTM.flatten(), 'Predictions': preds.flatten()})
        fpath_p = os.path.join("predictions",
                               "predictions_raw_fresh_" + datetime.now().strftime("%Y%m%d-%H%M%S") + ".csv")
        df.to_csv(fpath_p)

        return y_test_Conv_LSTM, preds, y_test_prep, preds_prep


    def generate(self, X_test_Conv_LSTM, length=1000, sample_range=(-1,1)):
        # ==============================Predicting on predicted=======================================
        Conv_LSTM_model = self.model
        seq_len = self.seq_length
        img_size = self.img_size

        gadf = GramianAngularField(image_size=img_size, method='difference', sample_range=sample_range)
        gasf = GramianAngularField(image_size=img_size, method='summation', sample_range=sample_range)
        mtf = MarkovTransitionField(image_size=img_size, n_bins=8, strategy='quantile')

        y_preds = []
        X_test = X_test_Conv_LSTM[0,:,:,:,:]
        for i in tqdm(range(length)):
            X_test = X_test.reshape(1, seq_len, img_size, img_size, 3)
            temp = Conv_LSTM_model.predict(X_test, verbose = 0, callbacks=None)
            y = temp
            y = y.reshape(1, -1)
            y_preds.append(y)
            last_two = X_test[0,1:,:,:,:]
            gadf_transformed = np.expand_dims(gadf.fit_transform(y), axis=3)
            gasf_transformed = np.expand_dims(gasf.fit_transform(y), axis=3)
            mtf_transformed = np.expand_dims(mtf.fit_transform(y), axis=3)

            prev_out = np.concatenate((gadf_transformed, gasf_transformed, mtf_transformed), axis=3)
            X_test = np.concatenate((last_two,prev_out), axis=0)

        y_preds = np.asarray(y_preds)
        y_pred = y_preds.reshape(-1)

        return y_pred
