import pyts
import numpy as np
import matplotlib.pyplot as plt

class ProcessWindows(object):
    def __init__(self, input_ts, window_size=2, stride=None, loglevel = 0):
        self.input_ts = name
        self.window_size = window_size
        self.stride = window_size if stride is None else stride
        self.loglevel = loglevel 

    def process_ts(self):
        serie = self.input_ts
        minx,maxx=min(serie),max(serie)
        r1,r2 = (-1,1)
        serie1=(r2-r1)*(serie-minx)/(maxx-minx)+r1
        #X=serie1.reshape(1,-1)
        self.normalized_ts = serie1
        Xw = windowmize(serie1,window_size=2, stride = 1)

        print("starting GramianAngularField") if self.loglevel > 0
        gasf = GramianAngularField(image_size=1., method='summation',sample_range=None)
        self.X_gasf = gasf.fit_transform(Xw)
        gadf = GramianAngularField(image_size=1., method='difference',sample_range=None)
        self.X_gadf = gadf.fit_transform(Xw)
        print("finished GramianAngularField") if self.loglevel > 0