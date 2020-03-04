# -*- coding: utf-8 -*-
"""
Holt Witer Class.
"""
import numpy as np
import matplotlib.pyplot as plt
import copy

class Holt_Winters_multiplicative:
    """"m is the seasonality, h the horizon of prediction, until now implementation 
        with h<m"""
    def __init__(self,serie,m,h,alpha,beta,gamma):
        self._serie=copy.deepcopy(serie)
        self._length=serie.size
        self._m=m
        self._h=h
        self._alpha=alpha
        self._beta=beta
        self._gamma=gamma
        
        
        self._l=np.zeros([self._length],dtype=float)
        self._b=np.zeros([self._length],dtype=float)
        self._s=np.zeros([self._length],dtype=float)
        """predicted contains simulated prediction for values of the series already
           given. It may be usefull for parameter tuning  or to assert the performance
           of the method. Anyway this is mainly in previson for the NN-Holt-Winter training """
        self._predicted=np.zeros([self._length+h],dtype=float)
        
        """from: https://robjhyndman.com/hyndsight/hw-initialization/"""
        self._lstart=serie[0:m].mean()
        self._bstart=sum((serie[m:2*m]-serie[0:m]))/(m**2)
        self._sstart=serie[0:m]/self._lstart
        
        
        
        
        self._l[0:self._m-1]=np.nan
        self._b[0:self._m-1]=np.nan
        self._predicted[0:self._m+self._h]=np.nan
        
        
        
        
        self._l[self._m-1]=self._lstart
        self._b[self._m-1]=self._bstart
        self._s[0:self._m]=self._sstart
        
    """ it computes the states l,b,s and does simulated prediction of values of the series."""
    """ it also computes the training error with the parameter alpha, beta and gamma """
        
    def compute_states(self):
        """Carefull: _l[0]=l(m-1), _b[0]=b(m-1), _s[0]=s(0), where the TS start from 0 """
        
      
        
        for t in range(self._m,self._length):
            self._l[t]=self._alpha*(self._serie[t]/self._s[t-self._m])+ \
                       (1-self._alpha)*(self._l[t-1]+self._b[t-1])
            self._b[t]=self._beta*(self._l[t]-self._l[t-1])+ \
                       (1-self._beta)*(self._b[t-1])
            self._s[t]=self._gamma*(self._serie[t]/(self._l[t-1]+self._b[t-1]))+ \
                       (1-self._gamma)*self._s[t-self._m]
                       
                    
            self._predicted[t+self._h]=(self._l[t]+self._h*self._b[t])*\
            self._s[t+self._h-self._m]
            
        self.MSE=((self._predicted[self._m+self._h:self._length]- \
                  self._serie[self._m+self._h:self._length])**2).mean()
        
        
     
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
            
    def parameter_refinment(self,refinmentLoops=5):
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
        plt.plot(range(0,self._length),self._serie,'r-')
        plt.plot(range(0,self._predicted.size),self._predicted,'b-')  
        
            
       
#%% demo and test of the class
      
"""https://stat.ethz.ch/Teaching/Datasets/airline.dat"""
"""did you know that ETH has public teahing dataset?"""
"""probaby there is a better way to read the table directly from the web"""

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

simulation=Holt_Winters_multiplicative(serie,m,h,alpha,beta,gamma)
simulation.compute_states()
simulation.print_predicted()
simulation.prediction()
simulation.parameter_refinment()

#%%
simulation.continuative_forecast(60)
simulation.print_predicted()

