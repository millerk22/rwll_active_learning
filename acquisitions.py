from graphlearning.active_learning import acquisition_function
import numpy as np
import logging


class uncnormprop_plusplus(acquisition_function):
    '''
    Sample proportional to the acquisition function's values.
    
    Currently implemented for SEQUENTIAL only
    '''
    def __init__(self):
        self.K = 10
        self.log_Eps_tilde = np.log(1e150)  # log of square root of roughly the max precision of python float
    
    def set_K(self, K):
        print(f"Setting K = {K} for uncnormprop++")
        self.K = K
        
    def compute(self, u, candidate_ind):
        # want to pick points with the highest uncertainty = smaller norm of u
        vals = 1. - np.linalg.norm(u[candidate_ind,:], axis=1)
        
        # scaling for p(x) \propto e^{x/T}, where T is scales as the values change. Ensures no numerical overflow occurs
        M = vals.max()
        T0 = M - np.percentile(vals, 100*(1. - 1./self.K))
        eps = M / (self.log_Eps_tilde - np.log(vals.size))
        T = max(eps, min(1.0,T0))
        p = np.exp(vals/T)
        
        # select a batch of points at random according to the probabilities previously calculated and then return standard basis vector of the maximizing index
        logging.debug(f"probabilities = {p}, p_sum = {p.sum()}")
        sample = min(10, candidate_ind.size)
        k_choices = np.random.choice(np.arange(candidate_ind.size), sample, replace=False, p=p/p.sum())
        k_choice = k_choices[np.argmax(vals[k_choices])]
        acq_vals = np.zeros_like(candidate_ind)
        acq_vals[k_choice] = 1.
        
        return acq_vals


class random(acquisition_function):
    '''
    Random choices
    '''
    def compute(self, u, candidate_ind):
        return np.random.rand(candidate_ind.size)
