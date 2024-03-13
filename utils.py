import numpy as np
from matplotlib import pyplot as plt
from scipy.io import loadmat
import os

def rescale(array): 
    '''
    Rescale by mean and std
    '''    
    m = np.mean(array)
    s = np.std(array)
    return (array-m)/s

def chop(trajs): 
    '''
    Chop up trajectories to the same lengths as the shortest one 
    '''
    L = min(map(len, trajs))
    new_trajs = [] 
    for traj in trajs: 
        start = 0 
        while start + L <= len(traj): 
            new_trajs.append(traj[start:start+L])
            start += L 
    return new_trajs


def find_corr(x, y):
    corr = np.correlate(x, y, mode='full')
    L = x.size
    n = np.arange(1, L+1)
    return corr[:L]/n


def find_auto_corr(x):
    corr = np.correlate(x, x, mode='full')
    L = x.size
    n = np.arange(1, L+1)
    return (corr[:L]/n)[::-1]