import numpy as np
import pandas as pd
import os
from datetime import datetime
from utils import sequence_splitter
from LSTM_callback import LSTM_callback

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Dropout, Flatten, LSTM, GlobalAveragePooling2D, BatchNormalization, TimeDistributed, Input, Bidirectional
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint

class noHW_LSTM:
    '''
    Pure LSTM class, used as baseline
    inp_window - equivalent
    '''
    def __init__(self,inp_window,out_window):
        self.inp_window=inp_window
        self.out_window=out_window
        self.model = self.build_NN()
        
    def build_NN(self):
        inp_window = self.inp_window
        out_window = self.out_window
            
        inp_shape=(inp_window,1)
        l_rate=0.001
        
        model = Sequential()
        model.add(LSTM(32, activation='relu',return_sequences=False ,input_shape=inp_shape))
        model.add(Dropout(0.5))
        model.add(Dense(out_window,activation='linear'))
        
        model.compile(RMSprop(learning_rate=l_rate),loss='mse', metrics=['mse'])
        model.summary()
        return model
         
    def fit_NN(self,train,test,bsize=32, epochs=5, p_filepath="predictions", l_filepath="tensorboard_logs", w_filepath="weights_LSTM", h=12, callbacks=True):

        from sklearn.preprocessing import MinMaxScaler
        MMscaler = MinMaxScaler(feature_range=(-1, 1))
        serie_train = MMscaler.fit_transform(train.reshape(-1, 1)).flatten()
        serie_test = MMscaler.fit_transform(test.reshape(-1, 1)).flatten()
        # serie_train = train.reshape(-1,1).flatten()
        # serie_test = test.reshape(-1,1).flatten()

        window_input_train, window_output_train = sequence_splitter(serie_train.reshape(-1, 1), self.inp_window, self.out_window, h)
        window_input_test, window_output_test = sequence_splitter(serie_test.reshape(-1, 1), self.inp_window, self.out_window, h)

        X_train = window_input_train
        y_train = window_output_train
        X_test = window_input_test
        y_test = window_output_test

        inp_window = self.inp_window
        out_window = self.out_window
        noHW_LSTM_model = self.model
        
        N_samples_train = np.shape(X_train)[0]
        N_samples_test = np.shape(X_test)[0]
        
        #Reshape Data
        X_train = X_train.reshape(N_samples_train,inp_window,1)
        y_train = y_train.reshape(N_samples_train,out_window,1)

        X_test = X_test.reshape(N_samples_test,inp_window,1)
        y_test = y_test.reshape(N_samples_test,out_window,1)
        # *** Callbacks *** #

        name = "LSTM_" + datetime.now().strftime("%Y%m%d-%H%M")
        logdir = os.path.join(l_filepath, name)
        tensorboard_callback = TensorBoard(log_dir=logdir)
        path_w = w_filepath
        filepath = os.path.join(path_w, "LSTM_weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5")
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath,
            monitor='loss',
            verbose=0,
            save_best_only=True,
            mode='min'
        )

        if callbacks:
            callback_list = [checkpoint, tensorboard_callback, LSTM_callback(validation_data=(X_test, y_test), scaler=MMscaler, logdir=logdir)]

        else:
            callback_list = []

        # *** Fit Model *** #
        history_noHW_LSTM = noHW_LSTM_model.fit(x=X_train, y=y_train, batch_size=bsize, epochs=epochs, callbacks=callback_list, shuffle=False, validation_data=(X_test,y_test))

        # *** Predict *** #
        print("Average test loss: ", np.average(history_noHW_LSTM.history['val_loss']))
            
        # *** Predict *** #    
        y_pred = noHW_LSTM_model.predict(X_test,verbose=1, callbacks=None)
        y_pred = y_pred.reshape(-1, 1)
        y_test = y_test.reshape(-1, 1)
        preds_unscaled = MMscaler.inverse_transform(y_pred)
        y_test_unscaled = MMscaler.inverse_transform(y_test)
        # preds_unscaled = y_pred
        # y_test_unscaled = y_test
        
        MSE = ((y_test_unscaled - preds_unscaled) ** 2).mean()
        df = pd.DataFrame({'True value': y_test_unscaled.flatten(), 'Predictions': preds_unscaled.flatten()})
        fname = "LSTM-" + datetime.now().strftime("%Y%m%d-%H%M")
        fpath_p = os.path.join(p_filepath, fname + ".csv")
        df.to_csv(fpath_p)
        
        return (history_noHW_LSTM,preds_unscaled,y_test_unscaled,MSE)
