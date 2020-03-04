from TSClass_function import *
from TSClass_lorenz import *
from TSClass_rossler import *
import math


##############################
"""
Example using TSClass_function
"""
def function_example(start_idx, end_idx, dt, sigma):
    """
    Example of callable object written the "correct way",
    that accepts as first arguments range of indexes [start_idx, end_idx)
    and return an array of (end_idx-start_idx,) values.
    In this case it is sin + noise
    """
    x1=start_idx*dt
    x2=(end_idx-1)*dt
    x=np.linspace(x1,x2,end_idx - start_idx)
    #assert (np.shape(x)[0]==)
    return np.sin(x)+np.random.normal(loc=0.0, scale=sigma, size=(end_idx - start_idx,))


def try_function_class():
    """
    Example of use of TSC_function. Here we save the values but the idea is that often we dont need
    the whole time series in memory, so the method get_next_N_vals gets the next N values
    and updates self.current_index to the new last position reached.
    """
    tsc1 = TSC_function(savevals=True)
    n=100
    dt=0.1
    data  = tsc1.get_next_N_vals(n,function_example, dt, sigma=0.1)
    data2 = tsc1.get_next_N_vals(n,function_example, dt, sigma=0.5)
    print(np.shape(tsc1.vals))
    tsc1.plot_range()



def try_lorenz():
    """
    Example of use of TSC_lorenz
    """
    tsc1 = TSC_lorenz(savevals=True)
    n=100
    dt=0.1
    data  = tsc1.get_next_N_vals(200)
    data2 = tsc1.get_next_N_vals(200)
#    data3 = tsc1.get_next_N_vals(10)
    print(np.shape(tsc1.vals))
    tsc1.plot_range()

def try_rossler():
    """
    Example of use of TSC_rossler
    """
    tsc1 = TSC_rossler(savevals=True,a = 0.38, b = 0.2, c = 5.7, coordinate='y')
    # other chaotic options
    # tsc2 = TSC_rossler(savevals=True,a = 0.38, b = 0.2, c = 5.7, coordinate='x')
    # tsc1 = TSC_rossler(savevals=True,a = 0.2, b = 0.2, c = 18,coordinate='x')
    
    n=100
    dt=0.1
    data  = tsc1.get_next_N_vals(200)
    data2 = tsc1.get_next_N_vals(2000)
#    data3 = tsc1.get_next_N_vals(10)
    print(np.shape(tsc1.vals))
    tsc1.plot_range()



#if __name__ == "__main__":
#try_function_class()
try_rossler()