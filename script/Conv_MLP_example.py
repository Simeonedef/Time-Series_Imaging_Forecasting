import matplotlib.pyplot as plt
from holt_winter import *
from TSClass_function import *
from TSClass_lorenz import *
from Conv_MLP import *
from utils import show_result

#%% Constants
img_size = 32


#%% Create Data

#Create Data
tsc3 = TSC_lorenz(savevals=True)
n=30000
dt=0.1
test_size= 10000
preds_filepath = "predictions_test" #Name of folder

serie  = 100+tsc3.get_next_N_vals(n)
serie_test=100+tsc3.get_next_N_vals(n)

print("=============TESTING HW============")
#%% Create Holt Winter object
print("Initializing Holt Winter")
HW=Holt_Winters_NN(serie,serie_test,m=1,h=2,windowsize=img_size,stride=1,alpha=0.25,beta=0,gamma=0.35,pr=1,compute_mtf=False)
#%% Call NN
print("Initializing Conv_MLP object")
model = Conv_MLP(img_size=img_size,N_Channel=2, test_size=test_size) #if compute_mtf=False -> set N_Channel=2
print("Starting Conv_MLP training")
history,y_pred,y_true,MSE = model.train_HW(HW, epochs=5, bsize=32, p_filepath=preds_filepath, callbacks=False)
print("Finished training")
print(MSE)
print("Showing results")
show_result(y_pred, y_true, title="Conv-MLP with HW", full=False)

print("=============TESTING RAW TS============")
print("Initializing Conv_MLP object")
model = Conv_MLP(img_size=img_size,N_Channel=3, raw=True, test_size=test_size) #if compute_mtf=False -> set N_Channel=2
print("Starting Conv_MLP training")
history,y_pred,y_true,MSE = model.train_series(serie[0:n],serie_test[0:test_size], epochs=5, bsize=32, p_filepath=preds_filepath,callbacks=False)
print("Finished training")
print(MSE)
print("Showing results")
show_result(y_pred, y_true, title="Conv-MLP without HW", full=False)
