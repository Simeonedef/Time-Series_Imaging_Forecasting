# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 17:37:48 2019

@author: 39389
"""

import numpy as np
import math
a=np.array([3,4,6,3,8,12]).reshape(3,2)
math.ceil(3.7)


window_size=2
stride = 1

N=np.shape(x)(0)
n_windows=math.ceil((N-window_size)/stride)

windoes=np.zeros([3,4])

