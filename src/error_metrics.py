import numpy as np
from math import sqrt

def minimax(err):
    return max(np.abs(err))

def mae(err):
    return sum(np.abs(err))/len(err)

def mse(err):
    return sum(np.abs(err)**2)/len(err)

def rmse(err):
    return sqrt(mse(err))