# -*- coding: utf-8 -*-
"""
Created on 2019
@author: Tangmei cgx@nuaa.edu.dn
Hypothesis transfer learning based on fuzzy residual
"""
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.kernel_ridge import KernelRidge
import skfuzzy as fuzz

def scatter_within(data):
    # mean_data 8191*8
    mean_data = np.tile(data.mean(0)[:, np.newaxis].T, ((data.shape[0], 1)))
    data_new = data - mean_data
    sum_norm = 0
    for i in range(data.shape[0]):
        sum_norm += np.linalg.norm(data_new[i,:])
    return sum_norm

def adjustScale(kernel_width, min_kernel, max_kernel):
    min_num = np.min(kernel_width)
    max_num = np.max(kernel_width)
    output = (kernel_width -min_num)/(max_num - min_num)
    output = min_kernel + output*((max_kernel - min_kernel))
    return output

def kernel_width(C, data, min_kernel, max_kernel):
    kernel_width = np.zeros((C.shape[0], data.shape[1]))
    for j in range(data.shape[1]):
        for k in range(C.shape[0]):
            kernel_width[k,j] = np.linalg.norm(data[:,j]
            - np.ones((data.shape[0]))*C[k,j])
    S = np.tile(kernel_width.sum(axis=0)[:, np.newaxis].T, ((kernel_width.shape[0], 1)))
    kernel_width = kernel_width/S
    kernel_width = adjustScale(kernel_width, min_kernel, max_kernel)
    return kernel_width
    
def gene_ante_deter(data, n_cluster):
    K = n_cluster
    clusters = np.ones(K).tolist()
    C = np.zeros((K, data.shape[-1]))
    k = 0
    while k < K:        
        class_ = [0,0]
        var_dimen = data.var(0)
        maxvar_index = np.argmax(var_dimen)
        data_maxvar = data[:, maxvar_index]
        mean_maxvar =np.mean(data_maxvar)
        class_[0] = data[data_maxvar<=mean_maxvar,:]
        class_[1] = data[data_maxvar>mean_maxvar,:]
        sum_norm_1 = scatter_within(class_[0])
        sum_norm_2 = scatter_within(class_[1])
        max_index_class = np.argmax([sum_norm_1,sum_norm_2])
        min_index_class = np.argmin([sum_norm_1,sum_norm_2])
        data = class_[max_index_class]
        clusters[k] = class_[min_index_class]
        k +=1
        if k==K-1:
            clusters[k] = class_[max_index_class]

    for i in range(K):
        C[i,:] = clusters[i].mean(0)
    
    b = kernel_width(C, data, 1, 10)
    return C, b
    
def fcm(data, n_cluster):
    """
    Comute data centers and membership of each point by FCM, and compute the variance of each feature
    :param data: n_Samples * n_Features
    :param n_cluster: number of center
    :return: centers: data center, delta: variance of each feature
    """
    n_samples, n_features = data.shape
    centers, mem, _, _, _, _, _ = fuzz.cmeans(
            data.T, n_cluster, 2.0, error=1e-5, maxiter=200)
    
    # compute delta compute the variance of each feature
    delta = np.zeros([n_cluster, n_features])
    for i in range(n_cluster):
        d = (data - centers[i, :]) ** 2
        delta[i, :] = np.sum(d * mem[i, :].reshape(-1, 1),
             axis=0) / np.sum(mem[i, :])
    return centers, delta

def get_x_p(data, centers, delta):
       
    """
    -- as euqation(11) in the paper
    Compute firing strength using Gaussian model
    :param data: n_Samples * n_Features
    :param centers: data center，n_Clusters * n_Features
    :param delta: variance of each feature， n_Clusters * n_Features
    :return: data_fs data: firing strength, 
        n_Samples * [n_Clusters * (n_Features+1)]
    """
#    delta = delta/3
    n_cluster = centers.shape[0]
    n_samples = data.shape[0]
    
    # compute firing strength of each data, n_Samples * n_Clusters
    mu_a = np.zeros([n_samples, n_cluster])
    for i in range(n_cluster):
        tmp_k = 0 - np.sum((data - centers[i, :]) ** 2 /
                           delta[i, :], axis=1)
        mu_a[:, i] = np.exp(tmp_k)  # exp max 709
        
    # norm
    mu_a = mu_a / np.sum(mu_a, axis=1, keepdims=True)
    
    # print(np.count_nonzero(mu_a!=mu_a))
    data_1 = np.concatenate((data, np.ones([n_samples, 1])), axis=1)
    
    zt = []
    for i in range(n_cluster):
        zt.append(data_1 * mu_a[:, i].reshape(-1, 1))
    data_fs = np.concatenate(zt, axis=1)
    data_fs = np.where(data_fs != data_fs, 1e-5, data_fs)
    return data_fs, mu_a

def model(X_source, X_target, Y_source, Y_target, X_test
          ,n_cluster=10, C=0.1, width = 1, residual = 'RD', basemodel='TSK',
          fit_with_target=True):
    n, d = X_source.shape

    """
    Build the prediction model of target domain with the help of source data.

    Parameters
    ----------
    X_source : X of source domain
    X_target : X of target domain
    y_source : X of source domain
    y_target : X of target domain
    X_test   : test data
    n_cluster: int 
        Number of clustering center
    c : float
        Regularization coefficient
    width : float
        The strenghthen of RD rules
    residual : string, "RD" or "LS"
        The bias can calculated by residual defuzzification "RD" or Least square method "LS".
    basemodel : string or callable, string 'TSK' or model from sklearn
        basemodel for dataset        
    fit_with_target: boolean
        fit the model with target data after label adaptation
        
    Returns
    -------
    Y_output : prediction results of test data
    x_each, y_each : for visualization of RD rules
    """
    # fuzzy partition methodself.label
    if d <= 3:  # fcm is good for low dimension
        centers, delta = fcm(X_source, n_cluster)
    else:      # varPart is good for high dimension
        centers, delta = gene_ante_deter(X_source, n_cluster)
    delta = delta*width  # change the rules width 
       
    # compute x_p: as euqation(3) in the paper
    X_p_s   , _  = get_x_p(X_source, centers, delta) # X_p_s as euqation(3)  
    X_p_t   , W  = get_x_p(X_target, centers, delta) # W as equation(20)
    X_p_test, _  = get_x_p(X_test,   centers, delta)
    
    # compute error of target data E
    if basemodel == 'TSK':       
        # compute consequent parametsrs by LS as equation(5) in the paper
        X_p_s1 = np.dot(X_p_s.T, X_p_s)
        Ps = np.linalg.pinv(X_p_s1 + C * np.eye(X_p_s1.shape[0])).dot(X_p_s.T).dot(Y_source)
        
        # compute generalization errors of source model on target data
        E = Y_target - X_p_t.dot(Ps)    
    else: # the basemodel from sklearn
        basemodel.fit(X_source, Y_source) 
        
        # compute generalization errors of source model on target data
        E = Y_target - basemodel.predict(X_target)
    
    # compute the bias Z by least squre (LS)
    if residual == 'LS':
        # equation(19)
        W_1 = np.dot(W.T, W)
        C2 = 1e-1       
        #  For those rules with no target data, the corresponding z_k will be controlled close to 0 
        #  with the help of the regularization term C2  
        Z = np.linalg.pinv(W_1 + C2 * np.eye(W_1.shape[0])).dot(W.T).dot(E)
     
     # compute the bias Z by residual defuzzification (RD)
    if residual == 'RD':   
        # equation(23) 
        Ws = np.sum(W, axis=0)
        W = W/Ws
        Z = W*E
        Z = np.sum(Z, axis=0)
                
    # the residual on X
    Pe = np.zeros((n_cluster*(d+1)))[:, np.newaxis]
    for k in range(n_cluster):
        Pe[k*(d+1)+d] = Z[k]
    residual_test   = X_p_test.dot(Pe)
    residual_source = X_p_s.dot(Pe)
    
    # cumpute the Y_output
    if fit_with_target == False:
        if basemodel == 'TSK': 
            Ytest_sourcemodel = X_p_test.dot(Ps)
            Y_output = Ytest_sourcemodel + residual_test
        else: # the basemodel from sklearn
            Ytest_sourcemodel = basemodel.predict(X_test)
            Y_output = Ytest_sourcemodel + residual_test
    else:
    # fit a output model with new data, anymodel is ok, here is KRR
        outmodel = GridSearchCV(KernelRidge(kernel='rbf', gamma=0.1), cv=5,
              param_grid={"alpha": [1e0, 0.1, 1e-2, 1e-3],
                          "gamma": np.logspace(-2, 2, 5)})
        if basemodel == 'TSK': 
            Ys_sourcemodel = X_p_s.dot(Ps)
            Ys_new = Ys_sourcemodel + residual_source
        else: # the basemodel from sklearn
            Ys_sourcemodel = basemodel.predict(X_source)
            Ys_new = Ys_sourcemodel + residual_source            
        X_all = np.vstack((X_source, X_target))
        Y_all = np.vstack((Ys_new, Y_target))
        outmodel.fit(X_all,Y_all)
        Y_output = outmodel.predict(X_test)

    # return  each rules, only when basemodel =='TSK'
    x_each = np.ones((3, n_cluster))
    y_each = np.ones((3, n_cluster))
    if basemodel == 'TSK': 
        for k in range(n_cluster):
            Ps[k*(d+1)+d] = Ps[k*(d+1)+d] + Z[k]
        dis = (max(X_source) - min(X_source))/(n_cluster*5)      
        for k in range(n_cluster):
            x_each[:,k] = [centers[k]-width*dis, centers[k], centers[k]+width*dis]        
            x_e = np.vstack((x_each[:,k].T, np.ones((3)))).T
            P_ =  Ps[k*(d+1): (k+1)*(d+1)]
            y_each[:,k] = x_e.dot(P_).reshape(-1)
               
    return Y_output, x_each, y_each
