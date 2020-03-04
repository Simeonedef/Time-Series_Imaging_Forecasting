import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import cv2

from holt_winter import *
from TSClass_function import *
from TSClass_lorenz import *
#%% Functions

def show_result(y, y_pred, full=False):
    plt.plot(y[150:750], c='red', lw=0.7)
    plt.plot(y_pred[150:750], c='blue',ls='--')
    plt.show()

    if full == True:
        plt.plot(y, c='red')
        plt.plot(y_pred, c='blue')
        plt.show()


#%% Constants
img_size = 32


#%% Create Data

#Create Data
tsc3 = TSC_lorenz(savevals=True)
tsc4 = TSC_lorenz(savevals=True)
n=30000
dt=0.1

serie  = 100+tsc3.get_next_N_vals(n)
serie_test=100+tsc4.get_next_N_vals(n)

#%% Create Holt Winter object
print("Initializing Holt Winter")
HW=Holt_Winters_NN(serie,serie_test,m=1,h=2,windowsize=img_size,stride=1,alpha=0.25,beta=0,gamma=0.35,pr=1,compute_mtf=False)

#%% Call NN
print("Initializing Conv_MLP object")
from Conv_MLP import *
model = Conv_MLP(img_size=img_size,N_Channel=2) #if compute_mtf=False -> set N_Channel=2


#%% Fit model
print("Starting Conv_MLP training")
history,y_pred,y_true,MSE = model.train_HW(HW,epochs=5, bsize=32)
# model.save("CONV_LSTM.hdf5")
print("Finished training")
print(MSE)


print("Showing results")
show_result(y_pred, y_true)



#%% Start Vizualization
''' See https://www.sicara.ai/blog/2019-08-28-interpretability-deep-learning-tensorflow'''
# assign model
MODEL = model.build_NN(img_size)

#Which image to observe?
ImageNumber = 1001 #Or sample number
img1=np.expand_dims(HW.gadf[ImageNumber],axis=2)
img2=np.expand_dims(HW.gasf[ImageNumber],axis=2)
img=np.concatenate((img1,img2),axis=2)

print('Original Images')
plt.imshow(img[:,:,0])
plt.show()
plt.imshow(img[:,:,1])
plt.show()

#Get layer names of model
Layer_name=[]
for layer in MODEL.layers:
    Layer_name.append(layer.name)


#%% Part I - Layer output vizualization of a given image
'''Inputs'''
layer_name =  Layer_name[5]
filter_index=6 # specify which filtered image to look at e.g. if shape=(N_samples,img,img,filter=64) -> filter_index<64
''' '''  

output = [layer.output for layer in MODEL.layers if layer.name in layer_name]

# Create a connection between the input and those target outputs
activations_model = tf.keras.models.Model(MODEL.inputs, outputs=output)
activations_model.compile(optimizer='adam', loss='mse')

# Get their outputs
activ = activations_model.predict(np.array([img]))

#Look at the output of a transformed image (transformed by filter_index filter)
plt.imshow(activ[0,:,:,filter_index])
plt.show()

#%% Part II - what makes a kernel activate? We give random noise and backpropagte this to the beginning
# '''Inputs'''
# layer_name =  Layer_name[1]
# filter_index = 12
# ''' '''

# epochs = 70         #Manual gradient ascent with epoch number
# step_size = 0.01
# N_channels=2



# # Create a connection between the input and the target layer
# submodel = tf.keras.models.Model([MODEL.inputs][0], [MODEL.get_layer(layer_name).output])

# # Initiate random noise
# input_img_data = np.random.random((1, 224, 224, N_channels))
# input_img_data = (input_img_data - 0.5) * 20 + 128.
# to_show_noise = input_img_data

# # Cast random noise from np.float64 to tf.float32 Variable
# input_img_data = tf.Variable(tf.cast(input_img_data, tf.float32))

# # Iterate gradient ascents
# for _ in range(epochs):
#     with tf.GradientTape() as tape:
#         outputs = submodel(input_img_data)
#         loss_value = tf.reduce_mean(outputs[:, :, :, filter_index])
#     grads = tape.gradient(loss_value, input_img_data)
#     normalized_grads = grads / (tf.sqrt(tf.reduce_mean(tf.square(grads))) + 1e-5)
#     input_img_data.assign_add(normalized_grads * step_size)



# to_show = input_img_data.numpy()
# plt.imshow(to_show_noise[0,:,:,0])
# plt.show()
# plt.imshow(to_show_noise[0,:,:,1])
# plt.show()
# plt.imshow(to_show[0,:,:,0])
# plt.show()
# plt.imshow(to_show[0,:,:,1])
# plt.show()

#%% Part III - Class Ativation Map -> Heatmap is mean of 2 channels...

layer_name =  Layer_name[0]



# Create a graph that outputs target convolution and output
grad_model = tf.keras.models.Model([MODEL.inputs], [MODEL.get_layer(layer_name).output, MODEL.output])


pred_index=0 #how many points to forecast -> Constant

# Get the score for target class
with tf.GradientTape() as tape:
    conv_outputs, predictions = grad_model(np.array([img])) #size:(1,img,img,filter) ; (1,1)
    loss = predictions[:, pred_index]

# Extract filters and gradients
output = conv_outputs[0]
grads = tape.gradient(loss, conv_outputs)[0]

# Average gradients spatially
weights = tf.reduce_mean(grads, axis=(0, 1))

# Build a ponderated map of filters according to gradients importance
cam = np.ones(output.shape[0:2], dtype=np.float32)

for index, w in enumerate(weights):
    cam += w * output[:, :, index]

aaa = cam.numpy()

# Heatmap visualization
cam = cv2.resize(cam.numpy(), (img_size, img_size))
cam = np.maximum(cam, 0)
heatmap = (cam - cam.min()) / (cam.max() - cam.min())

cam = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)


a1 = (img[:,:,0]+1)/2
b1 = (img[:,:,1]+1)/2

a2 =  np.uint8(255*a1)
b2 =  np.uint8(255*b1)

rgb_a2 = cv2.cvtColor(a2,cv2.COLOR_GRAY2RGB)
rgb_b2 = cv2.cvtColor(b2,cv2.COLOR_GRAY2RGB)


output_image_channel1 = cv2.addWeighted(rgb_a2, 1, cam, 0.5, 0)
output_image_channel2 = cv2.addWeighted(rgb_b2, 1, cam, 0.5, 0)


#Final Result
plt.imshow(img[:,:,0])
plt.show()
plt.imshow(img[:,:,1])
plt.show()
plt.imshow(cam)
plt.show()

