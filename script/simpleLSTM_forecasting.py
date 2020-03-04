'''
Simple LSTM Forecasting (multiwindow)
'''
#%%---------------------------------------------------------------------------
'''
Import Libraries
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Conv1D, MaxPooling1D, Dropout, Flatten, LSTM, GlobalAveragePooling1D, BatchNormalization, TimeDistributed
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import regularizers

#%%---------------------------------------------------------------------------
'''
Functions
'''
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def sequence_splitter(signal,inp_window,out_window):
    '''
    Split a signal into a X-Window and a Y-Window
    Returns:
        inp: input_window of shape(#samples,inp_window)
        out: output window of shape(#samples,out_window)
    The #samples is a function of the length of the signal, inp_window and out_window parameters.
    '''
    #Prepare windows to fill
    inp = list()
    out = list()
    
    for i in range(len(signal)):
        pointer_input = i+inp_window
        pointer_output = pointer_input + out_window
        
        if pointer_output > len(signal):
            break
            
        window_X = signal[i:pointer_input]
        window_Y = signal[pointer_input:pointer_output]
    
        inp.append(window_X)
        out.append(window_Y)
    
    # if show == True:
        # idx=10
    #     aid = np.arange(start=0,stop=len(signal))
    #     plt.plot(aid,signal,color='black',ls='--',lw=0.5)
    #     plt.plot(aid[inp_array[idx,:]],signal[inp_array[idx,:]],color='r',lw =3)
    #     plt.plot(aid[out_array[idx,:]],signal[out_array[idx,:]],color='b',lw=3)
    #     plt.show()     

    return np.array(inp), np.array(out)
        
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def show_result(y_test,y_pred):
    '''
    See a plot of the signal and the predicted signal
    '''
    y_test_show = np.ravel(y_test.reshape(N_samples,window_size_y))
    y_pred_show = np.ravel(y_pred.reshape(N_samples,window_size_y))
     
    plt.plot(y_test_show,ls='--',color='black')
    plt.plot(y_pred_show,color='red')
    plt.show()


#%%---------------------------------------------------------------------------
'''
Parameters
'''
time_points=200 #used to generate data
window_size_x = 10 #the x-window size
window_size_y = 2 #the y-window size
N_features = 1 #we have only one feature i.e. one type of time series


#%%---------------------------------------------------------------------------
'''
Data Generation
'''

#Test Set
x1 = np.linspace(-np.pi, np.pi, time_points)
sin1 = np.abs(np.sin(x1))
window_input_train,window_output_train = sequence_splitter(sin1,window_size_x,window_size_y)

#Train Set
x2 = np.linspace(-2*np.pi, 2*np.pi, time_points)
sin2=np.abs(np.sin(x2))
window_input_test,window_output_test = sequence_splitter(sin2,window_size_x,window_size_y)

if False:
    plt.plot(sin1)
    plt.plot(sin2)
    plt.show()
#%%---------------------------------------------------------------------------
'''
Data Preprocessing
'''
N_samples = np.shape(window_input_train)[0]

X_train = window_input_train.reshape(N_samples,window_size_x,N_features)
y_train = window_output_train.reshape(N_samples,window_size_y,N_features)

X_test = window_input_test.reshape(N_samples,window_size_x,N_features)
y_test = window_output_test.reshape(N_samples,window_size_y,N_features)

#%%---------------------------------------------------------------------------
'''
Simple LSTM Forecasting
'''

def NetworkCreator():
    
    
    model = Sequential()
    model.add(LSTM(50, activation='relu',return_sequences=False ,input_shape=(window_size_x, N_features)))
    # model.add(LSTM(50, activation='relu'))
    
    model.add(Dense(window_size_y,activation='sigmoid'))
    
    
    model.compile(optimizer='adam', loss='mse')
    return model

#Create Model
model = NetworkCreator()
print(model.summary())


#Fit Model  
history = model.fit(X_train, y_train, batch_size=4, epochs=30,use_multiprocessing=True,validation_data=(X_test,y_test))

#Prediction    
y_pred = model.predict(X_test)    

#Show Result
show_result(y_test,y_pred)
