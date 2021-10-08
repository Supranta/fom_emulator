import chaospy
import numpy as np
from sklearn.decomposition import PCA

def map2_unitinterval(samples, prior_lims):
    prior_upper = prior_lims[:,1]
    prior_lower = prior_lims[:,0]
    delta_prior = prior_upper - prior_lower
    return 2. * (samples - prior_lower) / delta_prior - 1.

class PCEEmulator:
    def __init__(self, N_DIM, prior_lims):
        self.N_DIM = N_DIM
        self.prior_lims = prior_lims    
    
    def train(self, fit_theta, scalar, N_ORDER=2):
        fit_x = map2_unitinterval(fit_theta, self.prior_lims)
        
        self.scalar_mean = np.mean(scalar, axis=0)
        self.scalar_std  = np.std(scalar, axis=0)
        scalar_norm = (scalar - self.scalar_mean) / self.scalar_std
        
        distribution = chaospy.J(chaospy.Uniform(-1., 1.), chaospy.Uniform(-1., 1.))
        for i in range(self.N_DIM - 2):
            distribution = chaospy.J(distribution, chaospy.Uniform(-1., 1.))        
        expansion = chaospy.generate_expansion(N_ORDER, distribution)
        solver_list = []
        
        solver = chaospy.fit_regression(expansion, fit_x.T, scalar_norm)
        
        self.solver = solver
        self.trained = True
        
    def predict(self, theta_pred):
        assert self.trained, "The emulator needs to be trained first before predicting"
        
        scalar_norm_pred_list = []
        pred_x = map2_unitinterval(theta_pred, self.prior_lims)
                
        scalar_pred = chaospy.call(self.solver, pred_x.T)
        scalar_pred = self.scalar_std * scalar_pred + self.scalar_mean
    
        return scalar_pred
         