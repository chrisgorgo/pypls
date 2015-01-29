import numpy as np
from pls import fit_pls, _procrustes_rotation
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

print "running procrustes on the same matrix"

rotated, rotation_matrix, rotated_singular_values = _procrustes_rotation(beh_saliences, beh_saliences, singular_values)
print "First rotatd beh. salience = %s"%rotated[:,0]
print " rotated singular_values = %s"%str(list(rotated_singular_values))
print "rotation matrix = %s"%str(rotation_matrix)

print "changing the beh data slightly and runnig PLS again"
beh_data[3:7] += 0.01
brain_data[30:70] -= 0.01

brain_saliences, new_beh_saliences, new_singular_values, inertia, brain_scaled, beh_scaled = fit_pls(brain_data, beh_data, n_components=n_components, scale=True, algorithm="randomized")

print "brain_saliences.shape = %s"%str(brain_saliences.shape)
print "beh_saliences.shape = %s"%str(new_beh_saliences.shape)
print "singular_values.shape = %s"%str(singular_values.shape)
print "singular_values = %s"%str(list(singular_values))
print "inertia = %s"%inertia
print "First beh. salience = %s"%new_beh_saliences[:,0]

print "running procrustes between the original and modified saliences"

rotated, rotation_matrix, rotated_singular_values = _procrustes_rotation(beh_saliences, new_beh_saliences, singular_values)
print "First rotatd beh. salience = %s"%rotated[:,0]
print " rotated singular_values = %s"%str(list(rotated_singular_values))
print "rotation matrix = %s"%str(rotation_matrix)