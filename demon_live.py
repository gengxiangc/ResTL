# -*- coding: utf-8 -*-
"""
Created on 2019
@author: Tangmei cgx@nuaa.edu.dn
Hypothesis transfer learning based on fuzzy residual
"""
import numpy as np
import matplotlib
matplotlib.rcParams['backend'] = 'SVG'
import matplotlib.pyplot as plt
import ResTL
from sklearn.model_selection import GridSearchCV
from sklearn.kernel_ridge import KernelRidge

''' 
CURVE : which demon curve 1, 2, 3 
RULES : visual the rules or not, only available when the  basemodel is TSK
nc    : the number of rules
C     : regularization 
width ：the width of rules
'''
CURVE = 2         
RULES = 1          
nc    = 6         
C     = 1e-8      
width = 1.5     

Title = 'Parameters: '+'nc='+str(nc)+' C='+str(C)+' width='+str(width)

# basemodel should be 'TSK' if you want to visual the basic rules
basemodel = GridSearchCV(KernelRidge(kernel='rbf', gamma=0.1), cv=5,
      param_grid={"alpha": [1e0, 0.1, 1e-2, 1e-3],
                  "gamma": np.logspace(-2, 2, 5)})

'''Demon curve 1
Xs, Ys: source data
Xt, Yt: target data
X_test：test data
'''
if CURVE == 1:    
    n_source1, n_target_train = 100, 5
    Xs_array = np.linspace(-10, 10, n_source1)
    Xs = Xs_array.reshape(len(Xs_array),1)
    Ys = np.cos(Xs[:,0])*Xs[:,0] + np.random.normal(0, 0.5, Xs.shape[0])
    Ys = Ys[:, np.newaxis]   
    X_test = np.linspace(-10, 10, 100)[:, np.newaxis]
    

'''simple curve 2'''
if CURVE == 2:  
    Xs = np.linspace(0, 1, 30)
    Xs = Xs.reshape(len(Xs),1)
    Ys = np.sin(7*Xs) + 1
    X_test = np.linspace(0, 1, 100)
    X_test = X_test.reshape(len(X_test),1)


def tellme(s):
    print(s)
    plt.title(s, fontsize=12)
    plt.draw()

'''Draw'''
fig, ax = plt.subplots()
tellme('Click to begin')
fontfamily = ['NSimSun'] 
font = {'family':fontfamily,
        'size':12,
        'weight':20}

Title = 'Parameters: '+'nc='+str(nc)+' C='+str(C)+' width='+str(width)
plt.waitforbuttonpress()
while True:
    plt.clf()
    plt.setp(plt.gca(), autoscale_on=True)
    if CURVE == 2:  
        plt.ylim([-2,3])
    if CURVE == 1:  
        plt.ylim([-20,15])

    plt.plot(Xs, Ys, 'r-', lw=2, label='Source model')
    pts = []
    tellme('Select target data with mouse, middle mouse button to finish')
    pts = np.asarray(plt.ginput(-1, timeout=-1))
    Xt, Yt = pts[:, 0].reshape((-1,1)), pts[:, 1].reshape((-1,1))
    
    ''' Predict the target model with propsed method'''
    y_pre, x_each, y_each = ResTL.model(Xs, Xt, Ys, Yt, X_test, 
                                         n_cluster=nc, 
                                         C=C, 
                                         width = width,
                                         residual = 'RD' , 
                                         basemodel = 'TSK',                                           
                                         fit_with_target=False)
      
    ''' visual the basic rules'''
    if RULES: 
        for i in range(nc):
            plt.plot(x_each[:,i], y_each[:,i], 'p-', lw=2)
            
    plt.plot(X_test, y_pre, 'g-', lw=2, zorder=9, label='ResTL')
    plt.plot(Xt, Yt, 'bo', lw=5, zorder=9, label='Target data')       
    plt.legend(prop=font)
    plt.xlabel(Title)
    
    tellme('Key click for over, mouse click for start again')
    
    if plt.waitforbuttonpress():
        break

tellme('Finished, you can close the window')
plt.show()
  
