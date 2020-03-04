"""
File containing TimeseriesClass
"""

from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt


class TimeseriesClass(ABC): # ABC=abstract class
    """
    TimeseriesClass
        Abstract base class to help with the timeseries management,
        can use just-in-time data generation/retrieval. Data are returned
        either as a (N,) dimensional np.array for 1D time series, or as a 
        (dims,N) array (rows=features, columns=datapoints) for dims>1.
    
    Fields:
        dims           (int)  : Feature dimension. If the time series is mono-dimensional,
                                it will be implemented as (N,) shaped array. If dims>1, it will be
                                implemented as (dims,N) array.
        savevals      (bool)  : True if we save the previous values of the timeseries.
        vals      (np.array)  : The array of previous values, if it exists.
        current_index  (int)  : The current index reached by the time series generated data.
                                If savevals is true it will be equivalent to the index of the last element (or of
                                the last column if dims>1).
    """

    def __init__(self, savevals = False, dims = 1):    
        """
        __init__: TimeseriesClass constructor

            Parameters:
            -- savevals (Bool): set to true to save all the obtained values in the object
            -- dims   (int): set to true to save all the obtained values in the object
        """
        self.savevals = savevals #bool
        self.current_index = 0 # Index of last element, equal to N-1
        self.dims = dims
        if savevals:
            if dims == 1:
                self.vals = np.empty((0,), dtype=np.float64, order='C')
            elif dims >1:
                self.vals = np.empty((0,0), dtype=np.float64, order='F') # F = Should be more cache efficient this way, for tipical access
            else:
                raise Exception("[ERROR][TimeseriesClass] dims should be either 1 or 2")
        super().__init__()

    @abstractmethod
    def get_next_N_vals(self, N, *args, **kwargs):
        """
        get_next_N_vals: get next N values from the timeseries. If dims is 1 it will return
                         a (N,) array, otherwise a (d,N) array. Abstract method for this class.
                         
            Parameters:
            -- N (int): How many elements to generate
        """
        pass


    def get_old_vals(self, i):
        """
        get_old_vals: if savedvals is true, will return previous values of the time series.
                         If dim = 1 will return self.vals[i], otherwise self.vals[:,i].
                         
            Parameters:
            -- i (int): return i-th element (zero based, as always).
        """
        if self.savevals:
            if self.dims == 1:
                return self.vals[i]
            else:
                return self.vals[:,i]
        else:
            raise Exception("[ERROR][TimeseriesClass] savevals == False, the previous values of \
                             the ts were not saved in this object!")
        
    def plot_range(self, start_idx = None, end_idx = None):
        """
        plot_range: simple plotting function to plot existing values, if saved.
                    Plot each feature separately. Plot the range [start_idx, end_idx).
                    Intended mostly for quick debug purposes, better to plot self.vals directly.
                    
            Parameters:
            -- start_idx (int): index of starting element (included), default=0.
            -- end_idx   (int): index of last element (excluded), default = current_index.
        """
        if self.savevals:
            if start_idx is None:
                start_idx = 0
            if end_idx is None:
                end_idx = self.current_index+1
            
            if self.dims == 1:
                plt.plot(self.vals[start_idx:end_idx])
            else:
                for dd in range(self.dims):
                    plt.plot(self.vals[dd,start_idx:end_idx],label=f'f: {dd}')
            plt.show()
        else:
            raise Exception("[ERROR][TimeseriesClass] savevals == False, the previous values of \
                            the ts were not saved in this object!")
    
    def _save(self, arr):
        """
        _save: If savevals is true, save the values in the internal array.
        """
        if self.savevals:
            if self.dims == 1:
                  # arr = (n,) array
                self.vals = np.r_[self.vals,arr]
                new_d = np.shape(arr)[0]
                self.current_index = self.current_index + new_d
            else: # arr = (d,n) array,append arr columns
                self.vals = np.c_[self.vals,arr]
                new_d = np.shape(arr)[1]
                self.current_index = self.current_index + new_d
        else:
            raise Exception("[ERROR][TimeseriesClass] savevals == False, not supposed to save data!")


            

