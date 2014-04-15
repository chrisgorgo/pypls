'''
Created on 15 Apr 2014

@author: gorgolewski
'''
import scipy as sp
import numpy as np
from sklearn.utils.extmath import randomized_svd

def _permute_and_calc_singular_values(X, Y, X_saliences, Y_saliences, singular_values_samples, perm_i, n_components):
    X_perm = np.random.permutation(X)
    covariance_perm = np.dot(Y.T, X_perm)
    Y_saliences_perm, singular_values_perm, X_saliences_perm = randomized_svd(covariance_perm, n_components=n_components)
    
    #It does not matter which side we use to calculate the rotated singular values
    #let's pick the smaller one for optimization
    if len(X_saliences_perm) > len(Y_saliences_perm):
        _, _, singular_values_samples[:,perm_i] = _procrustes_rotation(Y_saliences, Y_saliences_perm, singular_values_perm)
    else:
        _, _, singular_values_samples[:,perm_i] = _procrustes_rotation(X_saliences.T, X_saliences_perm.T, singular_values_perm)
    
    
def _boostrap(X, Y, X_saliences, Y_saliences, X_saliences_bootstraps, Y_saliences_bootstraps, bootstrap_i, n_components):
    sample_indices = np.random.choice(range(X.shape[0]), size=X.shape[0], replace=True)
    X_boot = X[sample_indices,:]
    Y_boot = Y[sample_indices,:]

    X_boot_scaled = sp.stats.mstats.zscore(X_boot, axis=0)
    Y_boot_scaled = sp.stats.mstats.zscore(Y_boot, axis=0)

    covariance_boot = np.dot(Y_boot_scaled.T, X_boot_scaled)
    Y_saliences_boot, _, X_saliences_boot = randomized_svd(covariance_boot, n_components=n_components)
    
    #It does not matter which side we use to calculate the rotated singular values
    #let's pick the smaller one for optimization
    if len(X_saliences_boot) > len(Y_saliences_boot):
        Y_saliences_bootstraps[:,:,bootstrap_i], rotation_matrix = _procrustes_rotation(Y_saliences, Y_saliences_boot)
        X_saliences_bootstraps[:,:,bootstrap_i] = np.dot(X_saliences_boot.T, rotation_matrix).T
    else:
        X_saliences_boot_rotated, _, rotation_matrix = _procrustes_rotation(X_saliences.T, X_saliences_boot.T)
        X_saliences_bootstraps[:,:,bootstrap_i] = X_saliences_boot_rotated.T
        Y_saliences_bootstraps[:,:,bootstrap_i] = np.dot(X_saliences_boot, rotation_matrix)

def _calc_p_val(singular_value_samples, singular_value, saliences_p_vals, component_i):
    saliences_p_vals[component_i] = (100-sp.stats.percentileofscore(singular_value_samples,singular_value))/100.0
    
def _procrustes_rotation(fixed, moving, moving_singular_values=None):
    N, _, P = np.linalg.svd(np.dot(fixed.T,moving))
    rotation_matrix = np.dot(N, P)
    rotated = np.dot(moving, rotation_matrix)
    
    if moving_singular_values:
        rotated_scaled = np.dot(np.dot(moving, np.diag(moving_singular_values)), rotation_matrix)
        rotated_singular_values = np.sqrt(np.square(rotated_scaled).sum(axis=0))
        return rotated, rotation_matrix, rotated_singular_values 
    else:
        return rotated, rotation_matrix
    

def PLS(X,Y, n_perm=1000, n_boot=1000, n_jobs=None):
        
    #scaling
    X_scaled = sp.stats.mstats.zscore(X, axis=0)
    Y_scaled = sp.stats.mstats.zscore(Y, axis=0)
    
    covariance = np.dot(Y_scaled.T, X_scaled)
    Y_saliences, singular_values, X_saliences = linalg.svd(covariance, full_matrices=False)
    inertia = singular_values.sum()
    
    n_components = len(singular_values)
    
    #permutation test
    singular_values_samples = np.zeros((n_components, n_perm))
    Parallel(n_jobs=n_jobs, backend="threading")(delayed(_permute_and_calc_singular_values)(X_scaled, Y_scaled, singular_values_samples, perm_i) for perm_i in range(n_perm))
    
    saliences_p_vals = np.zeros((n_components,))
    Parallel(n_jobs=n_jobs, backend="threading")(delayed(_calc_p_val)(singular_values_samples[component_i,:], singular_values[component_i], saliences_p_vals, component_i) for component_i in range(n_components))
       
    inertia_p_val = (100-sp.stats.percentileofscore(singular_values_samples.sum(axis=0), inertia))/100.0
    
    #bootstrap
    X_saliences_bootstraps = np.zeros(X_saliences.shape + (n_boot,))
    Y_saliences_bootstraps = np.zeros(Y_saliences.shape + (n_boot,))
    #for boot_i in range(n_boot):
    #    _boostrap(X, Y, X_saliences_bootstraps, Y_saliences_bootstraps, boot_i)
    #    print boot_i
    Parallel(n_jobs=n_jobs, backend="threading")(delayed(_boostrap)(X, Y, X_saliences_bootstraps, Y_saliences_bootstraps, boot_i) for boot_i in range(n_boot))
    
    X_saliences_bootstrap_ratios = X_saliences_bootstraps.mean(axis=2)/X_saliences_bootstraps.std(axis=2)
    Y_saliences_bootstrap_ratios = Y_saliences_bootstraps.mean(axis=2)/Y_saliences_bootstraps.std(axis=2)
    
    return X_saliences, Y_saliences, saliences_p_vals, X_saliences_bootstrap_ratios, Y_saliences_bootstrap_ratios, inertia, inertia_p_val, singular_values


def fit_pls(X, Y, n_components):
    #scaling
    X_scaled = sp.stats.mstats.zscore(X, axis=0)
    Y_scaled = sp.stats.mstats.zscore(Y, axis=0)
    
    covariance = np.dot(Y_scaled.T, X_scaled)
    Y_saliences, singular_values, X_saliences = randomized_svd(covariance, n_components=n_components)
    inertia = singular_values.sum()
    
    return X_saliences, Y_saliences, singular_values, inertia

def permutation_test(X_scaled, Y_scaled, X_saliences, Y_saliences, singular_values, inertia, n_perm):
    n_components = X_scaled.shape[0]
    singular_values_samples = np.zeros((n_components, n_perm))
    for perm_i in range(n_perm):
        _permute_and_calc_singular_values(X_scaled, Y_scaled, singular_values_samples, perm_i, n_components, X_saliences, Y_saliences) 
    
    saliences_p_vals = np.zeros((n_components,))
    for component_i in range(n_components):
        _calc_p_val(singular_values_samples[component_i,:], singular_values[component_i], saliences_p_vals, component_i) 
       
    inertia_p_val = (100-sp.stats.percentileofscore(singular_values_samples.sum(axis=0), inertia))/100.0
    
    return saliences_p_vals, inertia_p_val

        