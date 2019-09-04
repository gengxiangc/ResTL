# -*- coding: utf-8 -*-
"""
Created on 2019
@author: Tangmei cgx@nuaa.edu.dn
Hypothesis transfer learning based on fuzzy residual
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ResTL
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.metrics import mean_squared_error as mse
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'NSimSun'

kernel = 1.0 * RBF(length_scale=0.5, length_scale_bounds=(1e-2, 1e3)) \
    + WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-3, 1e+1))
  
'''
basemodel can be models from sklearn, such as GPR, KRR,
or basemodel can be 'TSK'
'''
basemodel = GaussianProcessRegressor(kernel=kernel,  
                              alpha=0.0) 

'''KIN'''
sourcedata = pd.read_csv('Data/kin-8fm.data',sep='\s+').values
targetdata = pd.read_csv('Data/kin-8nm.data',sep='\s+').values
Xs_ = sourcedata[:,0:8]
Ys_ = sourcedata[:,8][:, np.newaxis]
Xt_ = targetdata[:,0:8]
Yt_ = targetdata[:,8][:, np.newaxis]  
source_size, target_size = 0.03, 0.004


n_rules, C, width = 6, 1e-8, 1 

''' Res_TL_RD''' 
for i in range(20):
    rng = np.random.RandomState(i)
    
    Xs,Xs_test,Ys,Ys_test = train_test_split(Xs_,Ys_,test_size= 1 - source_size,random_state = i)     
    Xt,X_test,Yt,Y_test = train_test_split(Xt_,Yt_,test_size= 1 - target_size,random_state =i)
    
    y_pre,  x_each, y_each = ResTL.model(Xs, Xt, Ys, Yt, X_test, 
                                         n_cluster = n_rules, 
                                         C=C, 
                                         width = width,
                                         residual = 'RD', 
                                         basemodel = basemodel,                                           
                                         fit_with_target=True) 
    mes_ResTL = mse(Y_test ,y_pre) 
    print('MSE of ResTL  = ',mes_ResTL)