import numpy as np
import george
from george import kernels
from scipy.optimize import fmin_powell

def gp_ln_likelihood(hyper_pars, theta, scalar, ndim):  
    a = np.exp(hyper_pars[0])    
    b = np.exp(hyper_pars[1:]) 
    if (a < 0.) or (a > 20.):
        return np.inf
    for b_i in b:
        if (b_i > 1000.):
            return np.inf
    K = a * kernels.ExpSquaredKernel(b, ndim=ndim)
    gp = george.GP(K)
    gp.compute(theta)
    return -gp.lnlikelihood(scalar)

class GPEmulator:
    def __init__(self, N_DIM):
        self.N_DIM = N_DIM

    def train(self, fit_theta, scalar):
        init_hp_vals = np.ones(self.N_DIM + 1)
        
        # Normalize the scalars
        self.scalar_mean = np.mean(scalar)
        self.scalar_std  = np.std(scalar)
        self.scalar_norm = (scalar - self.scalar_mean) / self.scalar_std
        
        opt_hp_val = fmin_powell(gp_ln_likelihood, init_hp_vals, args=(fit_theta, self.scalar_norm, self.N_DIM))
        opt_hp_val = np.exp(opt_hp_val)       
        K  = opt_hp_val[0] * kernels.ExpSquaredKernel(opt_hp_val[1:], ndim=self.N_DIM)
        gp = george.GP(K)
        gp.compute(fit_theta)
        
        self.trained_gp = gp
        self.trained = True

    def predict(self, theta_pred):
        assert self.trained, "The emulator needs to be trained first before predicting"
        pca_pred_list = []
       
        scalar_pred = self.trained_gp.predict(self.scalar_norm, theta_pred, return_cov=False)
        scalar_pred = scalar_pred * self.scalar_std + self.scalar_mean
        
        return scalar_pred