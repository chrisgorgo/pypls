import numpy as np
from pls import fit_pls, permutation_test, bootstrap_test
from sklearn.cross_decomposition.pls_ import PLSSVD
n_subjects = 200
n_brain_dim = 2000
n_beh_dim = 15
n_components = 5

brain_data = np.random.rand(n_subjects, n_brain_dim)
beh_data = np.random.rand(n_subjects, n_beh_dim)

print "fitting the the PLS model"

brain_saliences, beh_saliences, singular_values, inertia, brain_scaled, beh_scaled = fit_pls(brain_data, beh_data, n_components=n_components, scale=True, algorithm="randomized")

print "brain_saliences.shape = %s"%str(brain_saliences.shape)
print "beh_saliences.shape = %s"%str(beh_saliences.shape)
print "singular_values.shape = %s"%str(singular_values.shape)
print "singular_values = %s"%str(list(singular_values))
print "inertia = %s"%inertia
print "First beh. salience = %s"%beh_saliences[:,0]

# pls_sk = PLSSVD(n_components=2, scale=True)
# pls_sk.fit(brain_data, beh_data)
# print pls_sk.y_weights_.T[0:2]
# 


print "fitting the same model with reversed order"

beh_saliences, brain_saliences, singular_values, inertia, _, _ = fit_pls(beh_data, brain_data, n_components=n_components, scale=True, algorithm="randomized")
print "brain_saliences.shape = %s"%str(brain_saliences.shape)
print "beh_saliences.shape = %s"%str(beh_saliences.shape)
print "singular_values.shape = %s"%str(singular_values.shape)
print "singular_values = %s"%str(list(singular_values))
print "inertia = %s"%inertia
print "First beh. salience = %s"%beh_saliences[:,0]

print "fitting the same model with ARPACK instead of randomized SVD"

brain_saliences, beh_saliences, singular_values, inertia, brain_scaled, beh_scaled = fit_pls(brain_data, beh_data, n_components=n_components, scale=True, algorithm="arpack")
print "brain_saliences.shape = %s"%str(brain_saliences.shape)
print "beh_saliences.shape = %s"%str(beh_saliences.shape)
print "singular_values.shape = %s"%str(singular_values.shape)
print "singular_values = %s"%str(list(singular_values))
print "inertia = %s"%inertia
print "First beh. salience = %s"%beh_saliences[:,0]

 
n_perm = 100
 
saliences_p_vals, inertia_p_val = permutation_test(brain_scaled, 
                                                   beh_scaled, 
                                                   brain_saliences, 
                                                   beh_saliences, 
                                                   singular_values, 
                                                   inertia, 
                                                   n_perm, 
                                                   verbose=True,
                                                   algorithm="arpack")
print "saliences_p_vals.shape = %s"%str(saliences_p_vals.shape)
print "saliences_p_vals = %s"%str(list(saliences_p_vals))
print "inertia_p_val = %s"%inertia_p_val

n_boot = 100

brain_saliences_bootstrap_ratios, beh_saliences_bootstrap_ratios = bootstrap_test(brain_data, beh_data, brain_saliences, beh_saliences, n_boot, n_components)

print "brain_saliences_bootstrap_ratios.shape = %s"%str(brain_saliences_bootstrap_ratios.shape)
print "max(brain_saliences_bootstrap_ratios) = %s"%str(brain_saliences_bootstrap_ratios.max())
print "beh_saliences_bootstrap_ratios.shape = %s"%str(beh_saliences_bootstrap_ratios.shape)
print "max(beh_saliences_bootstrap_ratios) = %s"%str(beh_saliences_bootstrap_ratios.max())
