# -*- coding: utf-8 -*-
from scipy.stats import chi2
import numpy as np

def fit_transit(t, y, p = 4, in_box = 2457518.4, out_box = 2457518.7):
    """
    This assumes the errors are all equal
    """
    A = np.zeros([p+2,p+2])
    b = np.zeros(p+2)
    idx = np.where((t>in_box)&(t<out_box))[0]
    # Fill solution matrix obtained with the derivate of the coefficients of 
    # the polynomials:
    for j in range(1,p+2):
        for i in range(1,p+2):
            A[j,i] = -np.sum(t**(j+i-2))
        A[j,0] = np.sum(t[idx]**(j-1))
        b[j] = np.sum((1.-y)*(t**(j-1)))

    # Fill solution matrix obtained with the derivative of delta:
    for i in range(p+1):
        A[0,i+1] = -np.sum(t[idx]**i)
    b[0] = np.sum(1.-y[idx])
    A[0,0] = np.double(len(idx))

    # Solve system:
    return np.linalg.solve(A,b)

def get_model(t, params, in_box = 2457518.4, out_box = 2457518.7):
    # Form model lightcurve:
    model = np.ones(len(t))
    for i in range(1,len(params)):
        model = model + params[i]*(t**(i-1))
    idx = np.where((t>in_box)&(t<out_box))[0]
    model[idx] = model[idx] - params[0]
    return model

def generate_kfolds(K,N):
    idx_training = []
    idx_validation = []
    n_in_validation = N/K
    all_idx = np.arange(N)
    c_idx = np.copy(all_idx)
    for i in range(K):
        c_idx_validation = np.random.choice(c_idx,n_in_validation,replace=False)
        c_idx_training = np.delete(all_idx,c_idx_validation)
        idx_training.append(np.copy(c_idx_training))
        idx_validation.append(np.copy(c_idx_validation))
        all_validation_idx = np.array([])
        for j in range(len(idx_validation)):
            all_validation_idx = np.append(all_validation_idx,idx_validation[j])
        c_idx = np.delete(all_idx,all_validation_idx)
        if i == K-2:
            n_in_validation = N-len(all_validation_idx)
    return idx_training,idx_validation
        

K = 10
t,f,ferr = np.loadtxt('datos.dat',unpack=True)
idx_training,idx_validation = generate_kfolds(K,len(t))
in_transit = 0.4
out_transit = 0.7
sigma = 30e-6
N = len(t)
pmax = 15
all_p = np.zeros(pmax)
all_cv = np.zeros(pmax)
all_cv_err = np.zeros(pmax)
print '\t  p     CVerr       Error'
print '\t ----------------------------'
import matplotlib.pyplot as plt
for p in range(pmax):
    cvs = np.array([])
    for k in range(K):
        params = fit_transit(t[idx_training[k]],f[idx_training[k]],\
                             p = p, in_box = in_transit, out_box = out_transit)
        model_validation = get_model(t[idx_validation[k]],params,in_box = in_transit, \
                                     out_box = out_transit)

        cvs = np.append(cvs,np.abs((model_validation-f[idx_validation[k]])*1e6))
    all_p[p] = p
    all_cv[p] = np.mean(cvs)
    all_cv_err[p] = np.sqrt(np.var(cvs)/len(cvs))
    print '\t'+r' {0:2} {1:10.3f} {2:10.3f}'.format(p,all_cv[p],all_cv_err[p])

import matplotlib.pyplot as plt
plt.style.use('ggplot')
plt.xlim([all_p[0]-0.5,all_p[-1]+0.5])
plt.errorbar(all_p,all_cv,yerr=all_cv_err,fmt='o-')
plt.ylabel('Cross-validation error')
plt.xlabel('$p$')
plt.show()
