import numpy as np
import pandas as pd
from datetime import datetime
import os

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Dropout, Flatten, LSTM, GlobalAveragePooling2D, BatchNormalization, TimeDistributed, Input, Bidirectional
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint

from utils import sequence_splitter
from MSE_HW_callback import *


class Conv_MLP:
    """
    This class will create a Conv_MLP network as well as do all the training provided the HW object
    Parameters to set:
    img_size - the size of the image supplied by the transformations

    """
    def __init__(self, img_size,N_Channel, raw=False, test_size=10000): # removed callback_path TF2.0 no support...
        self.img_size = img_size
        self.N_Channel = N_Channel
        self.test_size = test_size
        # self.callback_path = callback_path
        if(raw):
            self.model = None
        else:
            self.model = self.build_NN(img_size) #Builds automatically architecture when calling class!

    #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -         
    #Build Architecture    
    def build_NN(self,img_size,conv_layers=2,l_rate = 0.001):
        N_Channel = self.N_Channel
        
        #Parameters
        out_window=1
        Filters=8
        K_size = (5,5)
        #reg = regularizers.l1_l2(l1=0.01, l2=0.01)

        inp_shape = (img_size,img_size,N_Channel)
        
        model = Sequential()
        model.add(Conv2D(filters=Filters,kernel_size=K_size, activation='relu',input_shape = inp_shape))
        #model.add(BatchNormalization())
        #model.add(Dropout(0.5))
        
        
        
        model.add(Conv2D(filters=Filters,kernel_size=K_size,activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.1))
    
        model.add(Flatten())
        
        model.add(Dense(48, activation='relu'))
        #model.add(Dropout(0.5))
        model.add(Dense(32,activation='relu'))
        
        #model.add(Dense(16,activation='relu'))
        
        
        
        model.add(Dense(out_window,activation='linear'))
        
        model.compile(loss='mse', metrics=['mse'])
        model.summary()
        return model

    def Conv_MLP_series(self, img_size, window_size_y, N_channels_Conv_MLP):

        Filters = 8
        K_size = (5,5)
        inp_shape = (img_size, img_size, N_channels_Conv_MLP)
        conv_layers = 2

        model = Sequential()
        model.add(Conv2D(filters=Filters, kernel_size=K_size, activation='relu', input_shape=inp_shape))
        #model.add(BatchNormalization())
        #model.add(Dropout(0.5))
        
        
        model.add(Conv2D(filters=Filters,kernel_size=K_size,activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.1))

        
            

        model.add(Flatten())
        
        model.add(Dense(48, activation='relu'))
        #model.add(Dropout(0.5))
        model.add(Dense(32, activation='relu'))
        
        #model.add(Dense(16, activation='relu'))
        
        model.add(Dense(window_size_y, activation='linear'))

        model.compile(loss='mse', metrics=['mse'])
        return model
    
    #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -     
    #Train NN and predict    
    def train_HW(self, HW, epochs=5, bsize=32, p_filepath="predictions", l_filepath="tensorboard_logs", w_filepath="weights_conv_mlp", callbacks=True):
        Conv_MLP_model = self.model
        N_Channel = self.N_Channel
        
        # *** Train and Test sets *** #
        #Train set
        gadf_transformed_train = np.expand_dims(HW.gadf, axis=3)
        gasf_transformed_train = np.expand_dims(HW.gasf, axis=3)
        if N_Channel==3:    
            mtf_transformed_train = np.expand_dims(HW.mtf, axis=3)
            
        if N_Channel==3:    
            X_train = np.concatenate((gadf_transformed_train, gasf_transformed_train, mtf_transformed_train),axis=3)
        else:
            X_train = np.concatenate((gadf_transformed_train, gasf_transformed_train),axis=3)
        y_train = HW.obtain_training_output()

        #Test set
        gadf_transformed_test = np.expand_dims(HW.gadf_test, axis=3)
        gasf_transformed_test = np.expand_dims(HW.gasf_test, axis=3)
        if N_Channel==3:     
            mtf_transformed_test = np.expand_dims(HW.mtf_test, axis=3)
            
        if N_Channel==3:
            X_test = np.concatenate((gadf_transformed_test, gasf_transformed_test, mtf_transformed_test),axis=3)[0:self.test_size]
        else:
            X_test = np.concatenate((gadf_transformed_test, gasf_transformed_test),axis=3)[0:self.test_size]
        y_test = HW.test_output[0:self.test_size]
        y_test_raw = HW.test_output_val[0:self.test_size]

        # *** Callbacks *** #
        name = "Conv-MLP_" + datetime.now().strftime("%Y%m%d-%H%M")
        logdir = os.path.join(l_filepath, name)
        tensorboard_callback = TensorBoard(log_dir=logdir)
        path_w = w_filepath
        filepath = os.path.join(path_w, "Conv_MLP-weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5")
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath,
            monitor='loss',
            verbose=0,
            save_best_only=True,
            mode='min'
        )
        if callbacks:
            callback_list = [checkpoint, tensorboard_callback, MSE_HW_callback(HW, self.test_size, lstm=False, validation_data=(X_test, y_test), logdir=logdir)]
        else:
            callback_list = []

        # *** Fit Model *** #
        history_Conv_MLP = Conv_MLP_model.fit(x=X_train, y=y_train, batch_size=bsize, epochs=epochs, callbacks=callback_list,shuffle=False,
                                              validation_data=(X_test, y_test_raw))
            
        # *** Predict *** #    
        y_pred = Conv_MLP_model.predict(X_test,verbose=1, callbacks=None)
        
        y_pred_flat = y_pred.reshape(-1)
        y_pred_flat2 = np.multiply(y_pred_flat,HW.forecast_multiplier[0:self.test_size])
        y_true = np.multiply(y_test_raw, HW.forecast_multiplier[0:self.test_size])
        
        MSE = ((y_true - y_pred_flat2) ** 2).mean()

        print("Test loss with HW multiplier: ", MSE)
        df = pd.DataFrame({'True value': y_true.flatten(), 'Predictions': y_pred_flat2.flatten()})
        fname = "Conv_MLP-" + datetime.now().strftime("%Y%m%d-%H%M")
        fpath_p = os.path.join(p_filepath, fname + ".csv")
        df.to_csv(fpath_p)

        return (history_Conv_MLP,y_pred_flat2, y_true, MSE)

    def train_series(self, signal_train, signal_test, window_size_x=100, window_size_y=1, epochs=5, bsize=32, p_filepath="predictions", l_filepath="tensorboard_logs", w_filepath="weights_conv_mlp", h=12, callbacks=True):
        Conv_MLP_model = self.Conv_MLP_series(self.img_size, window_size_y, self.N_Channel)
        # N_Channel = self.N_Channel
        img_size = self.img_size

        signal_train = signal_train.reshape(-1, 1)
        signal_test = signal_test.reshape(-1, 1)
        sample_range = (-1, 1)

        # Scaling
        # from sklearn.preprocessing import MinMaxScaler
        #
        # MMscaler = MinMaxScaler(feature_range=(-1,1))
        # MMscaler_test = MinMaxScaler(feature_range=(-1, 1))

        # signal_train_scaled = MMscaler.fit_transform(signal_train).flatten()
        # signal_test_scaled = MMscaler_test.fit_transform(signal_test).flatten()

        signal_train_scaled = signal_train.flatten()
        signal_test_scaled = signal_test.flatten()

        # Split Sequence
        window_input_train, window_output_train = sequence_splitter(signal_train_scaled, window_size_x, window_size_y, h)
        window_input_test, window_output_test = sequence_splitter(signal_test_scaled, window_size_x, window_size_y, h)

        # %%---------------------------------------------------------------------------
        '''
        Field transformations
        '''
        from pyts.image import GramianAngularField
        from pyts.image import MarkovTransitionField

        gadf = GramianAngularField(image_size=img_size, method='difference', sample_range=sample_range)
        gasf = GramianAngularField(image_size=img_size, method='summation', sample_range=sample_range)
        mtf = MarkovTransitionField(image_size=img_size, n_bins=8, strategy='quantile')

        gadf_transformed_train = np.expand_dims(gadf.fit_transform(window_input_train), axis=3)
        gasf_transformed_train = np.expand_dims(gasf.fit_transform(window_input_train), axis=3)
        mtf_transformed_train = np.expand_dims(mtf.fit_transform(window_input_train), axis=3)

        X_train_windowed = np.concatenate((gadf_transformed_train, gasf_transformed_train, mtf_transformed_train),
                                          axis=3)

        gadf_transformed_test = np.expand_dims(gadf.fit_transform(window_input_test), axis=3)
        gasf_transformed_test = np.expand_dims(gasf.fit_transform(window_input_test), axis=3)
        mtf_transformed_test = np.expand_dims(mtf.fit_transform(window_input_test), axis=3)

        X_test_windowed = np.concatenate((gadf_transformed_test, gasf_transformed_test, mtf_transformed_test), axis=3)

        # Data reshaping
        X_train_Conv_MLP = X_train_windowed
        y_train_Conv_MLP = window_output_train

        X_test_Conv_MLP = X_test_windowed
        y_test_Conv_MLP = window_output_test

        # *** Callbacks *** #
        name = "Conv-MLP_raw_" + datetime.now().strftime("%Y%m%d-%H%M")
        logdir = os.path.join(l_filepath, name)
        tensorboard_callback = TensorBoard(log_dir=logdir)
        path_w = w_filepath
        filepath = os.path.join(path_w, "Conv_MLP_raw-weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5")
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath,
            monitor='loss',
            verbose=0,
            save_best_only=True,
            mode='min'
        )
        if callbacks:
            callback_list = [checkpoint, tensorboard_callback]
        else:
            callback_list = []

        # *** Fit Model *** #
        history_Conv_MLP = Conv_MLP_model.fit(x=X_train_Conv_MLP, y=y_train_Conv_MLP, batch_size=bsize, epochs=epochs,
                                              callbacks=callback_list, shuffle=False,
                                              validation_data=(X_test_Conv_MLP, y_test_Conv_MLP))

        # *** Predict *** #
        print("Average test loss: ", np.average(history_Conv_MLP.history['val_loss']))

        preds_prep = Conv_MLP_model.predict(X_test_Conv_MLP)
        preds_prep = preds_prep.reshape(-1, 1)
        y_test_prep = y_test_Conv_MLP.reshape(-1, 1)
        # preds_unscaled = MMscaler_test.inverse_transform(preds_prep)
        # y_test_unscaled = MMscaler_test.inverse_transform(y_test_prep)
        preds_unscaled = preds_prep
        y_test_unscaled = y_test_prep
        MSE_test_no_HW = ((y_test_unscaled - preds_unscaled) ** 2).mean()
        print("Test loss: ", MSE_test_no_HW)
        # show_result(y_test_prep, preds_prep)

        df = pd.DataFrame({'True value': y_test_unscaled.flatten(), 'Predictions': preds_unscaled.flatten()})
        fname = "Conv_MLP_raw-" + datetime.now().strftime("%Y%m%d-%H%M")
        fpath_p = os.path.join(p_filepath, fname + ".csv")
        df.to_csv(fpath_p)

        return (history_Conv_MLP, preds_unscaled, y_test_unscaled, MSE_test_no_HW)