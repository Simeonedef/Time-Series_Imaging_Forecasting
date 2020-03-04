# -*- coding: utf-8 -*-
"""
Holt Witer Classes
Stefano d'Apolito
"""
import numpy as np
import copy
import pyts
from windowmize import *
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from pyts.image import GramianAngularField
from pyts.image import MarkovTransitionField
from pyts.datasets import load_gunpoint



#This class implement the Holt Winte classifier (multiplicative version) It predicts h time steps ahead given the test previous values of te series
#
class Holt_Winters_multiplicative:
    
    def __init__(self,serie,serie_test,m,h,alpha,beta,gamma):
        """
           Parameters:
               -- serie: the training serie
               -- serie_test: the test serie (to simplify the implementation it must be of the same length of serie)
               -- m: seasonality of the serie
               -- h: horizon of prediction
               -- alpha,beta,gamma: intial guess of parameters of HW. The optimal
                   parameters are computed by the method parameter refinment given the training serie
          
           Requires: if m!=1, m>h (i.e. prediction are possible only within a season)"""    
    
        #initialising values
        self._serie=copy.deepcopy(serie)
        self._serie_test=copy.deepcopy(serie_test)
        self._length=serie.size
        self._m=m
        self._h=h
        self._alpha=alpha
        self._beta=beta
        self._gamma=gamma         
        
        #l,s,b are the state of Holt Winter
        self._l=np.zeros([self._length],dtype=float)
        self._b=np.zeros([self._length],dtype=float)
        
        if self._m!=1:
            self._s=np.zeros([self._length],dtype=float)
        else:
            self._s=np.ones([self._length+self._h],dtype=float)
            
        self._l_test=np.zeros([self._length],dtype=float)
        self._b_test=np.zeros([self._length],dtype=float)
        #if the seasonality is 1 i.e. no seasonality, put all s to 1
        if self._m!=1:
            self._s_test=np.zeros([self._length],dtype=float)
        else:
            self._s_test=np.ones([self._length+self._h],dtype=float)
        
        self._predicted=np.zeros([self._length+h],dtype=float)
        self._predicted_test=np.zeros([self._length+h],dtype=float)
        
        #from: https://robjhyndman.com/hyndsight/hw-initialization/ 
          #the inizialization method of s,l,b was taken from the linked site
        #inizialization of s,l,b
        self._lstart=serie[0:m].mean()
        self._bstart=sum((serie[m:2*m]-serie[0:m]))/(m**2)
        self._sstart=serie[0:m]/self._lstart
        
        self._lstart_test=serie_test[0:m].mean()
        self._bstart_test=sum((serie_test[m:2*m]-serie_test[0:m]))/(m**2)
        self._sstart_test=serie_test[0:m]/self._lstart_test 
        
        
        self._l[0:self._m-1]=np.nan
        self._b[0:self._m-1]=np.nan
        self._predicted[0:self._m+self._h]=np.nan
        self._predicted_test[0:self._m+self._h]=np.nan
        
        self._l_test[0:self._m-1]=np.nan
        self._b_test[0:self._m-1]=np.nan
        
        
        self._l[self._m-1]=self._lstart
        self._b[self._m-1]=self._bstart
        self._s[0:self._m]=self._sstart
        
        self._l_test[self._m-1]=self._lstart_test
        self._b_test[self._m-1]=self._bstart_test
        self._s_test[0:self._m]=self._sstart_test
        
    
        
    def compute_states(self):
        """ it computes the states l,b,s 
            it also computes the training error with the parameter alpha, beta and gamma """
        
        for t in range(self._m,self._length):
            self._l[t]=self._alpha*(self._serie[t]/self._s[t-self._m])+ \
                       (1-self._alpha)*(self._l[t-1]+self._b[t-1])
            self._b[t]=self._beta*(self._l[t]-self._l[t-1])+ \
                       (1-self._beta)*(self._b[t-1])
            if(self._m!=1):
                self._s[t]=self._gamma*(self._serie[t]/(self._l[t-1]+self._b[t-1]))+ \
                           (1-self._gamma)*self._s[t-self._m]
            
                       
                    
            self._predicted[t+self._h]=(self._l[t]+self._h*self._b[t])*\
            self._s[t+self._h-self._m]
            
        self.MSE=((self._predicted[self._m+self._h:self._length]- \
                  self._serie[self._m+self._h:self._length])**2).mean()
        
    def compute_states_test(self):
        """ it computes the states l,b,s for the test series and it does the actual prdiction"""
        
        for t in range(self._m,self._length):
            self._l_test[t]=self._alpha*(self._serie_test[t]/self._s_test[t-self._m])+ \
                       (1-self._alpha)*(self._l_test[t-1]+self._b_test[t-1])
            self._b_test[t]=self._beta*(self._l_test[t]-self._l_test[t-1])+ \
                       (1-self._beta)*(self._b[t-1])
            if(self._m!=1):
                self._s_test[t]=self._gamma*(self._serie_test[t]/(self._l_test[t-1]+self._b_test[t-1]))+ \
                           (1-self._gamma)*self._s_test[t-self._m]
            
                
                       
                    
            self._predicted_test[t+self._h]=(self._l_test[t]+self._h*self._b_test[t])*\
            self._s_test[t+self._h-self._m]
            
        self.MSE=((self._predicted_test[self._m+self._h:self._length]- \
                  self._serie_test[self._m+self._h:self._length])**2).mean()
        
        [n_windows,lag] = windomize_size(self._length-self._h-self._m,self._windowsize,self._stride)
        
        return self._predicted_test[self._m+lag+self._windowsize-1+self._h:self._length]
        
        
     
    def prediction(self):
        return(self._predicted[self._length+h-1:-1])
        
    def continuative_forecast(self,window):
        for t in range(window):
            
            tmp=self._alpha*(self._predicted[self._length+t]/ \
                      self._s[self._length+t-self._m])+(1-self._alpha)* \
                      (self._l[self._length+t-1]+self._b[self._length+t-1])
            self._l=np.append(self._l,tmp)
            
            tmp=self._beta*(self._l[self._length+t]-self._l[self._length+t-1])+ \
                       (1-self._beta)*(self._b[self._length+t-1])
            self._b=np.append(self._b,tmp)
            
            tmp=self._gamma*(self._predicted[self._length+t]/ \
                     (self._l[self._length+t-1]+self._b[self._length+t-1]))+ \
                     (1-self._gamma)*self._s[self._length+t-self._m]
            self._s=np.append(self._s,tmp)
                       
             
            tmp=(self._l[self._length+t]+self._h*\
                 self._b[self._length+t])*self._s[self._length+t+self._h-self._m]
            self._predicted=np.append(self._predicted,tmp)
            
    def parameter_refinment(self,refinmentLoops=4):
        """it optimizes the parameter alpha beta gamma given the training series. 
        It try to find the best combination of coefficients seeing the one that provide the lowest training error"""
        
        alpha_search=np.linspace(0,0.5,50)
        beta_search=np.linspace(0,0.2,50)
        gamma_search=np.linspace(0,0.5,50)
        
        best_alpha=self._alpha
        best_beta=self._beta
        best_gamma=self._gamma
        
        best_MSE=self.MSE
        
        print('initial MSE:',best_MSE)
        
        for i in range(refinmentLoops):
            
            for alpha in alpha_search:
                self._alpha=alpha
                self.compute_states()
                
                if(self.MSE<best_MSE):
                    best_MSE=self.MSE
                    best_alpha=self._alpha
            self._alpha=best_alpha
            
            for beta in beta_search:
                self._beta=beta
                self.compute_states()
                if(self.MSE<best_MSE):
                    best_MSE=self.MSE
                    best_beta=self._beta
            self._beta=best_beta
            
            for gamma in gamma_search:
                self._gamma=gamma
                self.compute_states()
                if(self.MSE<best_MSE):
                    best_MSE=self.MSE
                    best_gamma=self._gamma
            self._gamma=best_gamma
            self.MSE=best_MSE
            print('MSE:',self.MSE)       
                
        
        
    def print_predicted(self):
        plt.figure(figsize=(8,8))
        plt.plot(range(0,self._length),self._serie,'r-')
        plt.plot(range(0,self._predicted.size),self._predicted,'b-')  
        

class Holt_Winters_NN(Holt_Winters_multiplicative):
    """this class produce anything to train and test the NN"""    
    def __init__(self,serie,serie_test,m,h,windowsize=12,stride=1,alpha=0.25,beta=0,gamma=0.35,pr=3,compute_mtf=True):
        """Parameters:
           -- serie: the training serie
           -- serie_test: the test serie (to simplify the implementation it must be of the same length of serie)
           -- m: seasonality of the serie
           -- h: horizon of prediction
           -- alpha,beta,gamma: intial guess of parameters of HW. The optimal
               parameters are computed by the method parameter refinment given the training serie
           -- windowsize: the size of the window 
           --stride: every how many steps a prediction is done e.g. stride=2 a predition is done a time t,an other a time t+2, predicting t+h, and t+h+2
           -- compute_mtf wheter computing the mtf field: the library pyts does not manage to compute this field for Lorenz series
          Requires: if m!=1, m>h (i.e. prediction are possible only within a season)"""
        super(Holt_Winters_NN,self).__init__(serie,serie_test,m,h,alpha,0,gamma)
        self._b[self._m-1]=0
        self._b_test[self._m-1]=0
        self.compute_states()
        self.parameter_refinment(pr)
        self.compute_states_test()     
        

        #the vector to give to the NN for training (i.e. the time series scaled and desonalised)                   
        self._training_vector=(self._serie[self._m:self._length-self._h]/ \
                              self._l[self._m:(self._length-self._h)])/ \
                              self._s[0:(self._length-self._h-self._m)]
        self._test_vector=(self._serie_test[self._m:self._length-self._h]/ \
                              self._l_test[self._m:(self._length-self._h)])/ \
                              self._s_test[0:(self._length-self._h-self._m)]
                              
                              
        
        self._windowsize=windowsize
        self._stride=stride 
        #n_windows=length of the list of images,lag: the first lag element of the serie are not used so that the windowsize fit the length of the serie
        [n_windows,lag] = windomize_size(self._training_vector.size,self._windowsize,self._stride)
        #serie deseasonalised and scaled, from which obtaining the imgs to give to the NN
        self._training_output=self._serie[self._m+lag+self._windowsize-1+self._h:self._length]/ \
                              (self._l[(self._m+lag+self._windowsize-1):(self._length-self._h)]* \
                              self._s[(lag+self._windowsize-1+self._h):(self._length-self._m)]) 
        #value for which the prediction of the NN must be multiplied                      
        self.forecast_multiplier=self._l_test[(self._m+lag+self._windowsize-1):(self._length-self._h)]* \
                              self._s_test[(lag+self._windowsize-1+self._h):(self._length-self._m)]
        #contains the value of the test serie aligned with the prediction                      
        self.test_output=self._serie_test[self._m+lag+self._windowsize-1+self._h:self._length]
        self.test_output_val=self._serie_test[self._m+lag+self._windowsize-1+self._h:self._length]/ \
                              (self._l_test[(self._m+lag+self._windowsize-1):(self._length-self._h)]* \
                              self._s_test[(lag+self._windowsize-1+self._h):(self._length-self._m)])
                              
        #self._training_output_multiple=np.zeros([m,self._training_output.size])
        
        #check end of the vector it may 
        #for hh in range(1,self._m+1):
            #self._training_output_multiple[hh-1,:]=self._serie[self._m+lag+self._windowsize-1+hh:self._length]/ \
                              #(self._l[(self._m+lag+self._windowsize-1):(self._length-hh)]* \
                              #self._s[(lag+self._windowsize-1):(self._length-hh-self._m)]) 
                              
        

        print(self._training_vector.mean())
        
        #computation of the list of images for training and test
        b=max(self._training_vector)
        a=min(self._training_vector)
        
        self._scale=b-a
        self._training_vector=2*(self._training_vector-a)/(b-a)-1
        
        b=max(self._test_vector)
        a=min(self._test_vector)
        
        self._scale_test=b-a
        self._test_vector=2*(self._test_vector-a)/(b-a)-1
        
        self._training_matrix=windowmize(self._training_vector,self._windowsize,self._stride)
        gasf = GramianAngularField(image_size=1., method='summation',sample_range=None)
        self.gasf = gasf.fit_transform(self._training_matrix)
        gadf = GramianAngularField(image_size=1., method='difference',sample_range=None)
        self.gadf = gadf.fit_transform(self._training_matrix)
    
        if(compute_mtf):
            mtf=MarkovTransitionField(image_size=1.)
            self.mtf= mtf.fit_transform(self._training_matrix)
        
        #in case of a first dense layer they could be usefull
        #self.concatenated_images=np.concatenate((self.gadf,self.gasf), axis=1)
        #self.concatenated_images=np.concatenate((self.gadf,self.gasf,self.mtf), axis=1)
        
        self._test_matrix=windowmize(self._test_vector,self._windowsize,self._stride)
        gasf_test = GramianAngularField(image_size=1., method='summation',sample_range=None)
        self.gasf_test = gasf_test.fit_transform(self._test_matrix)
        gadf_test= GramianAngularField(image_size=1., method='difference',sample_range=None)
        self.gadf_test= gadf_test.fit_transform(self._test_matrix)
        #check if it is correct
        if(compute_mtf):
            mtf_test=MarkovTransitionField(image_size=1.)
            self.mtf_test= mtf_test.fit_transform(self._test_matrix)
        
        #self.concatenated_images_test=np.concatenate((self.gadf_test,self.gasf_test), axis=1)
        #self.concatenated_images_test=np.concatenate((self.gadf_test,self.gasf_test,self.mtf_test), axis=1)
        


        
        #plt.figure(figsize=(8,8))                     
        #plt.plot(self._training_vector)
        
        
    def obtain_training_output(self):
        return copy.deepcopy(self._training_output)
    #they do not work, just take the field 
    def obtain_gasf(self):
        return copy.deepcopy(self.gasf)
    
    def obtain_gadf(self):
        return copy.deepcopy(self.gadf)
    
    def obtain_mtf(self):
        return copy.deepcopy(self.matf)
    
        
        
        
    def compute_states(self):
        """it computes the states of HW for the training serie"""
        for t in range(self._m,self._length):
            self._l[t]=self._alpha*(self._serie[t]/self._s[t-self._m])+ \
                       (1-self._alpha)*(self._l[t-1]+self._b[t-1])
            self._b[t]=0
            if self._m!=1:
                self._s[t]=self._gamma*(self._serie[t]/(self._l[t-1]+self._b[t-1]))+ \
                           (1-self._gamma)*self._s[t-self._m]
                       
                    
            self._predicted[t+self._h]=(self._l[t]+self._h*self._b[t])*\
            self._s[t+self._h-self._m]
            
        self.MSE=((self._predicted[self._m+self._h:self._length]- \
                  self._serie[self._m+self._h:self._length])**2).mean()
        
    def compute_states_test(self):
        """it computes the states of HW for the test series"""
        for t in range(self._m,self._length):
            self._l_test[t]=self._alpha*(self._serie_test[t]/self._s_test[t-self._m])+ \
                       (1-self._alpha)*(self._l_test[t-1]+self._b_test[t-1])
            self._b_test[t]=0
            if self._m!=1:
                self._s_test[t]=self._gamma*(self._serie_test[t]/(self._l_test[t-1]+self._b_test[t-1]))+ \
                           (1-self._gamma)*self._s_test[t-self._m]
                       
                    
            
        
    
    def parameter_refinment(self,refinmentLoops=2):
        """it computes the optimimal parameter of HW alpha and gamma, given the training serie.
        Parameters:
            pr=number of loop iteration in order to ind the best combination of parameter """
        alpha_search=np.linspace(0,0.5,50)
        gamma_search=np.linspace(0,0.5,50)
        
        best_alpha=self._alpha
        best_gamma=self._gamma
        
        best_MSE=self.MSE
        
        print('refining parameter of HW, initial training MSE:',best_MSE)
        
        for i in range(refinmentLoops):
            
            for alpha in alpha_search:
                self._alpha=alpha
                self.compute_states()
                
                if(self.MSE<best_MSE):
                    best_MSE=self.MSE
                    best_alpha=self._alpha
            self._alpha=best_alpha
            
            
            for gamma in gamma_search:
                self._gamma=gamma
                self.compute_states()
                if(self.MSE<best_MSE):
                    best_MSE=self.MSE
                    best_gamma=self._gamma
            self._gamma=best_gamma
            self.MSE=best_MSE
            print('MSE:',self.MSE)
    
    def continuative_forecast(self,window):
        """see the demo"""
        for t in range(window):
            
            tmp=self._alpha*(self._predicted[self._length+t]/ \
                      self._s[self._length+t-self._m])+(1-self._alpha)* \
                      (self._l[self._length+t-1]+self._b[self._length+t-1])
            self._l=np.append(self._l,tmp)
            
            self._b=np.append(self._b,0)
            
            tmp=self._gamma*(self._predicted[self._length+t]/ \
                     (self._l[self._length+t-1]+self._b[self._length+t-1]))+ \
                     (1-self._gamma)*self._s[self._length+t-self._m]
            self._s=np.append(self._s,tmp)
                       
             
            tmp=(self._l[self._length+t]+self._h*\
                 self._b[self._length+t])*self._s[self._length+t+self._h-self._m]
            self._predicted=np.append(self._predicted,tmp)
            
    def print_predicted(self):
        plt.figure(figsize=(8,8))
        plt.plot(range(0,self._length),self._serie,'r-')
        plt.plot(range(0,self._predicted.size),self._predicted,'b-')
        #plt.figure(figsize=(8,8))
        #plt.plot(range(0,self._length),self._serie/self._predicted[0:144],'r-')
        
        
        
        
     
            
    
     
       
#%% little demo and test of the class
if __name__ == "__main__":
    """https://stat.ethz.ch/Teaching/Datasets/airline.dat"""
    """airline traffic"""
    
    serie=np.array([112, 118, 132, 129, 121, 135, 148, 148, 136, 119, 104, 118, 
                    115, 126, 141, 135, 125, 149, 170, 170, 158, 133, 114, 140, 
                    145, 150, 178, 163, 172, 178, 199, 199, 184, 162, 146, 166, 
                    171, 180, 193, 181, 183, 218, 230, 242, 209, 191, 172, 194, 
                    196, 196, 236, 235, 229, 243, 264, 272, 237, 211, 180, 201, 
                    204, 188, 235, 227, 234, 264, 302, 293, 259, 229, 203, 229, 
                    242, 233, 267, 269, 270, 315, 364, 347, 312, 274, 237, 278, 
                    284, 277, 317, 313, 318, 374, 413, 405, 355, 306, 271, 306, 
                    315, 301, 356, 348, 355, 422, 465, 467, 404, 347, 305, 336, 
                    340, 318, 362, 348, 363, 435, 491, 505, 404, 359, 310, 337, 
                    360, 342, 406, 396, 420, 472, 548, 559, 463, 407, 362, 405, 
                    417, 391, 419, 461, 472, 535, 622, 606, 508, 461, 390, 432])
            
    alpha=0.25
    beta=0.002
    gamma=0.35
    m=12  #12 month#
    h=6   #6 month prediction#
    stride=1
    windowsize=12
    #%%
    simulation=Holt_Winters_multiplicative(serie,serie,m,h,alpha,beta,gamma)
    simulation.compute_states()
    simulation.print_predicted()
    simulation.parameter_refinment()
        
    
    #%%
    print('qua')
    simulation.continuative_forecast(60)
    simulation.print_predicted()
    simulation.prediction()
    #%%  HW_NN test
    simulation2=Holt_Winters_NN(serie,serie,12,h,stride=1,alpha=0.25,beta=0,gamma=0.35)
    simulation2.continuative_forecast(60)
    simulation2.print_predicted()
    
