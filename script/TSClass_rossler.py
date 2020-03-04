import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from TimeseriesClass import *

class TSC_rossler(TimeseriesClass): # ABC=abstract class
    """
    TSC_function
        Class derived from TimeseriesClass, that generates data from a x-projection of
        a Rossler attractor.

        Fields:
            dims           (int)     : Feature dimension. If the time series is mono-dimensional,
            a,b,c            (float) : Parameters of the Rossler attractor (see Wikipedia).
            last_state (float array) : (3,) XYZ array representing the current total state.
            last_time  (float)       : Time of last state
            deltat     (float)       : Time step size
    """

    def __init__(self, a = 0.38, b = 0.2, c = 5.7, 
                 state0 = [0.0, 0.0, 0.0], coordinate='x', start_time = 0, deltat=0.1, savevals = False, dims = 1):  
        """
        __init__: TSC_function constructor

            Parameters:
            -- rho, sigma, beta (float): set Lorenz parameters
            -- state0 ((3,) float array): set initial state
            -- start_time (float): set initial time
            -- deltat: set time step
            -- savevals (Bool): set to true to save all the obtained values in the object
            -- dims   (int): set to true to save all the obtained values in the object
        """
        self.a = a
        self.b = b
        self.c = c
        self.last_state = state0
        self.last_time = start_time
        self.deltat = deltat
        coordid = {'x': 0, 'y': 1, 'z': 2}
        self.coord = coordid[coordinate]

        super().__init__(savevals, dims)

    def get_next_N_vals(self, N):
        """
        get_next_N_vals: get next N values from the timeseries. Will return
                         a (N,) array representing the X-projection of the state.
                         
            Parameters:
            -- N (int): How many elements to generate
        """
        dt = self.deltat
        ti = self.last_time
        tf = self.last_time + N*dt
        statei = self.last_state

        times = np.linspace(ti, tf, N+1)

        states = odeint(self._derivatives, statei, times)

        # update initial state
        self.last_time = times[-1]
        self.last_state = states[-1,:]

        d = states[1:, self.coord]
        if self.savevals:
            self._save(d)
        return d
    
    def _derivatives(self, state, t):
        """
        _derivatives: The 3D gradient vector field of the Lorenz ODEs
        """
        x, y, z = state  # Unpack the state vector
        return -y-z, x+self.a*y, self.b+z*(x-self.c) # Derivatives
