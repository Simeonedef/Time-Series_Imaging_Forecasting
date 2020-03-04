import numpy as np
import matplotlib.pyplot as plt
from TimeseriesClass import *
import pandas as pd



class TSC_function(TimeseriesClass): # ABC=abstract class
    """
    TSC_function
        Class derived from TimeseriesClass, that generates data from a supplied function or lambda, that should be
        of the form f(start_idx, end_idx, *args, **kwargs) and return either a (N,) array, otherwise a (d,N) array, with
        N = end_idx-start_idx. The range is usually interpreted as [start_idx, end_idx).
    """

    def __init__(self, savevals = False, dims = 1):    
        """
        __init__: TSC_function constructor

            Parameters:
            -- savevals (Bool): set to true to save all the obtained values in the object
            -- dims   (int): set to true to save all the obtained values in the object
        """
        super().__init__(savevals, dims)

    def get_next_N_vals(self, N, f, *args, **kwargs):
        """
        get_next_N_vals: get next N values from the timeseries. If dims is 1 it will return
                         a (N,) array, otherwise a (d,N) array. Abstract method for this class.
                         
                         This method will use the callable f as f(N,*args, **kwargs) to generate 
                         the data, expecting the output to conform to the above convention.
                         
            Parameters:
            -- N (int): How many elements to generate
            -- f (callable): Callable object to generate data, will be called as f(cur_idx+1, cur_idx+N, *args, **kwargs)
                             and is expected to return either a (N,) array or a (dims,N) array.
            -- *args, **kwargs: additional parameters for f
        """
        d = f(self.current_index, self.current_index + N, *args, **kwargs)
        if self.savevals:
            self._save(d)
        return d