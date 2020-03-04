#*****
''' Visualization Lorenz Conv-MLP with HW'''
#****


import numpy as np
import matplotlib.pyplot as plt
import h5py
from tensorflow.keras.models import load_model

from Conv_MLP import *
from holt_winter import *
from TSClass_function import *
from TSClass_lorenz import *
from Visualization_Conv_MLP_2Channel import *

#%% Load Model

#Parameters    
n = 200000
dt = 0.1
img_size = 40
preds_filepath="predictions_lorenz"
w_filepath = "weights_lorenz"
l_filepath = "logs_lorenz"
seq_len = 3
out_window = 1
test_size = 10000
epochs = 10
bsize = 32
h = 10
in_window = 40

#Model
model_init = Conv_MLP(img_size=img_size,N_Channel=2, test_size=test_size)
model = model_init.build_NN(img_size=img_size)

#Load Weights
path_w = ("weights_lorenz")
filepath = os.path.join(path_w, "Conv_MLP-weights-0.0030-bigger.hdf5")
model.load_weights(filepath)


#%% Generate Images from HW class to investigate

#Generate time series
np.random.seed(57)
tsc = TSC_lorenz(savevals=True)
serie  = 100+tsc.get_next_N_vals(n)

#Call HW
HW=Holt_Winters_NN(serie,serie,m=1,h=h,windowsize=img_size,stride=1,alpha=0.35,beta=0.15,gamma=0.1)


#Images
gadf_transformed_train = np.expand_dims(HW.gadf, axis=3)
gasf_transformed_train = np.expand_dims(HW.gasf, axis=3)
X_train = np.concatenate((gadf_transformed_train, gasf_transformed_train),axis=3)

#%%
'''
Start Vizualizing
'''

#Choose image to investigate
#************************* Choose Image
img_index = 9562
#**************************
img_gadf = X_train[img_index,:,:,0]
plt.imshow(img_gadf)
plt.show()
img_gasf = X_train[img_index,:,:,1]
plt.imshow(img_gasf)
plt.show()
img = X_train[img_index]

plt.imshow(img_gadf)
plt.show()
#%%

#Initialize
Vis = visualization(model)

#Get layer names
lay_names = Vis.get_layer_names(show=True)

#Look at shapes
lay_shapes = Vis.get_layer_shapes(show=True)

#Chose image, which define above
Vis.set_image(img)

#******************** Choose which layer. Look at "Get layer names". No dense layer visualization
layer_idx = 0
#********************
#Set layer index 
lay_index = Vis.set_layer_index(layer_idx)


#Get output layer
Output_vis = Vis.out_layer_vis(show=True)

#Get activity map
Heat_vis = Vis.activity_map_vis(show=True)
