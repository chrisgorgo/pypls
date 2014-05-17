'''
Created on 15 Apr 2014

@author: gorgolewski
'''
import scipy as sp
import numpy as np
from sklearn.utils.extmath import randomized_svd
from scipy.stats.mstats_basic import zscore
import pyprind
from sklearn.utils import arpack
from sklearn.decomposition.truncated_svd import TruncatedSVD

def _permute_and_calc_singular_values(X, Y, X_saliences, Y_saliences, singular_values_samples, perm_i, n_components, procrustes=False, algorithm="randomized"):
    if len(X) < len(Y):
        X_perm = np.random.permutation(X)
        covariance_perm = np.dot(Y.T, X_perm)
    else:
        Y_perm = np.random.permutation(Y)
        covariance_perm = np.dot(Y_perm.T, X)
    svd = TruncatedSVD(n_components, algorithm=algorithm)
    Y_saliences_perm, singular_values_perm, X_saliences_perm =  svd._fit(covariance_perm)
    
    if procrustes:
    #It does not matter which side we use to calculate the rotated singular values
    #let's pick the smaller one for optimization
        if len(X_saliences_perm) > len(Y_saliences_perm):
            _, _, singular_values_samples[:,perm_i] = _procrustes_rotation(Y_saliences, Y_saliences_perm, singular_values_perm)
        else:
            X_saliences_perm = X_saliences_perm.T
            _, _, singular_values_samples[:,perm_i] = _procrustes_rotation(X_saliences, X_saliences_perm, singular_values_perm)
    else:
        singular_values_samples[:,perm_i] = singular_values_perm
    
    
def _boostrap(X, Y, X_saliences, Y_saliences, X_saliences_bootstraps, Y_saliences_bootstraps, bootstrap_i, n_components, algorithm="randomized"):
    sample_indices = np.random.choice(range(X.shape[0]), size=X.shape[0], replace=True)
    X_boot = X[sample_indices,:]
    Y_boot = Y[sample_indices,:]

    X_boot_scaled = sp.stats.mstats.zscore(X_boot, axis=0, ddof=1)
    Y_boot_scaled = sp.stats.mstats.zscore(Y_boot, axis=0, ddof=1)

    covariance_boot = np.dot(Y_boot_scaled.T, X_boot_scaled)
    svd = TruncatedSVD(n_components, algorithm=algorithm)
    Y_saliences_boot, _, X_saliences_boot = svd._fit(covariance_boot)
    X_saliences_boot = X_saliences_boot.T
    
    #It does not matter which side we use to calculate the rotated singular values
    #let's pick the smaller one for optimization
    if len(X_saliences_boot) > len(Y_saliences_boot):
        Y_saliences_bootstraps[:,:,bootstrap_i], rotation_matrix = _procrustes_rotation(Y_saliences, Y_saliences_boot)
        X_saliences_bootstraps[:,:,bootstrap_i] = np.dot(X_saliences_boot, rotation_matrix)
    else:
        X_saliences_boot_rotated, _, rotation_matrix = _procrustes_rotation(X_saliences, X_saliences_boot)
        X_saliences_bootstraps[:,:,bootstrap_i] = X_saliences_boot_rotated
        Y_saliences_bootstraps[:,:,bootstrap_i] = np.dot(X_saliences_boot, rotation_matrix)

def _calc_p_val(singular_value_samples, singular_value, saliences_p_vals, component_i):
    saliences_p_vals[component_i] = (100-sp.stats.percentileofscore(singular_value_samples,singular_value))/100.0
    
def _procrustes_rotation(fixed, moving, moving_singular_values=None):
    N, _, P = np.linalg.svd(np.dot(fixed.T,moving))
    rotation_matrix = np.dot(N, P)
    rotated = np.dot(moving, rotation_matrix)
    
    if moving_singular_values != None:
        rotated_scaled = np.dot(np.dot(moving, np.diag(moving_singular_values)), rotation_matrix)
        rotated_singular_values = np.sqrt(np.square(rotated_scaled).sum(axis=0))
        return rotated, rotation_matrix, rotated_singular_values 
    else:
        return rotated, rotation_matrix
    

def PLS(X,Y, n_perm=1000, n_boot=1000, n_jobs=None):
        
    #scaling
    X_scaled = sp.stats.mstats.zscore(X, axis=0, ddof=1)
    Y_scaled = sp.stats.mstats.zscore(Y, axis=0, ddof=1)
    
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


def fit_pls(X, Y, n_components, scale=True, algorithm="randomized"):
    #scaling
    if scale:
        X_scaled = zscore(X, axis=0, ddof=1)
        Y_scaled = zscore(Y, axis=0, ddof=1)
        covariance = np.dot(Y_scaled.T, X_scaled)
    else:
        covariance = np.dot(Y.T, X)

    svd = TruncatedSVD(n_components, algorithm)
    Y_saliences, singular_values, X_saliences = svd._fit(covariance)
    X_saliences = X_saliences.T
    inertia = singular_values.sum()
    
    if scale:
        return X_saliences, Y_saliences, singular_values, inertia, X_scaled, Y_scaled
    else:
        return X_saliences, Y_saliences, singular_values, inertia

def permutation_test(X_scaled, Y_scaled, X_saliences, Y_saliences, singular_values, inertia, n_perm, verbose=False, algorithm="randomized"):
    n_components = X_saliences.shape[1]
    singular_values_samples = np.zeros((n_components, n_perm))
    
    if verbose:
        my_perc = pyprind.ProgBar(n_perm, stream=1, title='running permutations', monitor=True)
        #import warnings
        #warnings.filterwarnings("ignore")
    for perm_i in range(n_perm):
        _permute_and_calc_singular_values(X_scaled, Y_scaled, X_saliences, Y_saliences, singular_values_samples, perm_i, n_components, algorithm=algorithm)
        if verbose:
            my_perc.update()
    if verbose:
        print my_perc
        print "calculating p values"
    
    saliences_p_vals = np.zeros((n_components,))
    for component_i in range(n_components):
        saliences_p_vals[component_i] = (100.0-sp.stats.percentileofscore(singular_values_samples[component_i,:], singular_values[component_i]))/100.0
       
    inertia_p_val = (100.0-sp.stats.percentileofscore(singular_values_samples.sum(axis=0), inertia))/100.0
    
    return saliences_p_vals, inertia_p_val

def bootstrap_test(X, Y, X_saliences, Y_saliences, n_boot, n_components):
    #bootstrap
    X_saliences_bootstraps = np.zeros(X_saliences.shape + (n_boot,))
    Y_saliences_bootstraps = np.zeros(Y_saliences.shape + (n_boot,))
    #for boot_i in range(n_boot):
    #    _boostrap(X, Y, X_saliences_bootstraps, Y_saliences_bootstraps, boot_i)
    #    print boot_i
    for boot_i in range(n_boot):
        _boostrap(X, Y, X_saliences, Y_saliences, X_saliences_bootstraps, Y_saliences_bootstraps, boot_i, n_components)
    
    X_saliences_bootstrap_ratios = X_saliences_bootstraps.mean(axis=2)/X_saliences_bootstraps.std(axis=2)
    Y_saliences_bootstrap_ratios = Y_saliences_bootstraps.mean(axis=2)/Y_saliences_bootstraps.std(axis=2)
    
    return X_saliences_bootstrap_ratios, Y_saliences_bootstrap_ratios
        