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
CURVE = 3         
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
    Ys = np.cos(Xs[:,0])*Xs[:,0] + np.random.normal(0, 0.15, Xs.shape[0])
    Ys = Ys[:, np.newaxis]
    
    Xt = np.linspace(-10, -2, n_target_train)
    Xt = Xt.reshape(len(Xt),1)
    Yt = np.cos(Xt[:,0])*Xt[:,0] + Xt[:,0] + np.random.normal(0, 0.15, Xt.shape[0])
    Yt = Yt[:, np.newaxis]
    
    X_test = np.linspace(-10, 10, 100)[:, np.newaxis]

    

'''Demon curve 2'''
if CURVE == 2:  
    n_source1, n_target_train = 100, 8
    
    Xs_array = np.linspace(-1, 1, n_source1)
    Xs = Xs_array.reshape(len(Xs_array),1)
    Ys = np.cos(10*Xs[:,0])*Xs[:,0] + np.random.normal(0, 0.0085, Xs.shape[0])
    Ys = Ys[:, np.newaxis]
    
    Xt = np.linspace(-0.4, 1, n_target_train)
    Xt = Xt.reshape(len(Xt),1)
    Yt = np.cos(10*Xt[:,0])*Xt[:,0] + Xt[:,0] + np.random.normal(0, 0.0085, Xt.shape[0])  +1 
    Yt = Yt[:, np.newaxis]
    
    X_test = np.linspace(-1, 1, 100)[:, np.newaxis]


'''Demon curve 3'''
if CURVE == 3:  
    Xs = np.linspace(0, 1, 30)
    Xs = Xs.reshape(len(Xs),1)
    Ys = np.sin(7*Xs) + 1
#    Ys = Ys[:, np.newaxis]
        
    Xt =  np.linspace(0.5, 0.8, 3)
    Xt = Xt.reshape(len(Xt),1)
    Yt = Xt*np.sin(7*Xt) + 0.1
#    Yt = Yt[:, np.newaxis]
    
    X_test = np.linspace(0, 1, 100)
    X_test = X_test.reshape(len(X_test),1)
    


'''main'''
Y_RD_KRR,  x_each, y_each = ResTL.model(Xs, Xt, Ys, Yt, X_test, 
                                         n_cluster=nc, 
                                         C=C, 
                                         width = width,
                                         residual = 'RD' , 
                                         basemodel = basemodel,                                           
                                         fit_with_target=False)

Y_LS_TSK,  x_each, y_each = ResTL.model(Xs, Xt, Ys, Yt, X_test, 
                                         n_cluster=nc, 
                                         C=C, 
                                         width = width,
                                         residual = 'LS' , 
                                         basemodel = 'TSK',                                           
                                         fit_with_target=False)

'''fig'''
fig, ax = plt.subplots()
plt.plot(X_test, Y_LS_TSK, 'purple', lw=2, zorder=9, label='ResTL$_{LS}$') 
plt.plot(X_test, Y_RD_KRR, 'g', lw=2, zorder=9, label='ResTL$_{RD}$')  

'''visual rules'''
if RULES:
    for i in range(nc):
        plt.plot(x_each[:,i], y_each[:,i], 'p-', lw=2)
        
plt.plot(Xs, Ys, 'r-', lw=3, label='Source model')
plt.scatter(Xt, Yt, c='b', s=100, label='Target Data')
fontfamily = 'NSimSun'
font = {'family':fontfamily,
        'size':12,
        'weight':23}

ax.set_xlabel('X',fontproperties = fontfamily, size = 12)
ax.set_ylabel('Y',fontproperties = fontfamily, size = 12)
plt.yticks(fontproperties = fontfamily, size = 12) 
plt.xticks(fontproperties = fontfamily, size = 12) 
ax.set_title(Title, fontproperties = fontfamily, size = 12)
plt.legend(prop=font)
plt.tight_layout()
plt.ylim([-2,3])
plt.legend(prop=font)
plt.show()



