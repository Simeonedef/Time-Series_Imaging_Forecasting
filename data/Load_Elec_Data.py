'''
Descritpion:
    The DataFrame "Data" has clients and the 105216 (=1096 days = 3years)
    of datapoints each (time). 
    
    One datapoint = 15min
    4 datapoints = 1h
    96 datapoints = 1 day

    It makes sense, that we chose our seasonality as 1day = 96 datapoints.
    See https://archive.ics.uci.edu/ml/datasets/ElectricityLoadDiagrams20112014
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#%%
'''
Loader & Sampler
'''
Data = pd.read_csv('Elec_Data_Processed.csv', sep=",")      #Load Data
Data.drop(['Unnamed: 0'],axis=1,inplace=True)               #Indexer before

resampl = 1
Data_res = Data[::resampl]                              #resample data if needed

#%%
'''
Plotting:
    The DataFrame Data has 321 clients and the 105216 (=1096 days = 3years)
    of datapoints each (time). 
    
    One datapoint = 15min
    4 datapoints = 1h
    96 datapoints = 1 day

    It makes sense, that we chose our seasonality as 1day = 96 datapoints.

'''

#5000:5500,24
B=Data.iloc[5000:5500,28]

plt.figure()
plt.plot(B,linestyle='--')
Data_description= Data.describe()


#%%


