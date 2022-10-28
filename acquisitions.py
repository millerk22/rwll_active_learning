from graphlearning.active_learning import acquisition_function, model_change, v_opt, model_change_vopt, uncertainty_sampling, sigma_opt
import numpy as np
from scipy.special import softmax
from scipy.stats import entropy as spentropy
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances




class uncsftmax(acquisition_function):
    '''
    Smallest margin on softmax of the output, u
    '''
    def compute_values(self, active_learning, u):
        s = softmax(u, axis=1)
        u_sort = np.sort(s)
        return 1. - (u_sort[active_learning.candidate_inds,-1] - u_sort[active_learning.candidate_inds,-2]) # smallest margin

class uncdist(acquisition_function):
    '''
    Straightforward Euclidean distance to current pseudolabel
    '''
    def compute_values(self, active_learning, u):
        one_hot_predicted_labels = np.eye(u.shape[1])[np.argmax(u, axis=1)]
        return  np.linalg.norm((u - one_hot_predicted_labels), axis=1)[active_learning.candidate_inds]


class uncnorm(acquisition_function):
    '''
    Norm of the output rows in u.
    '''
    def compute_values(self, active_learning, u):
        return 1. - np.linalg.norm(u[active_learning.candidate_inds], axis=1)

class random(acquisition_function):
    '''
    Random choices
    '''
    def compute_values(self, active_learning, u):
        return np.random.rand(u.shape[0])[active_learning.candidate_inds]

    
class static_values(acquisition_function):
    '''
    Return static values of the graph (e.g. PageRank, Degree, LAND)
    '''
    def compute_values(self, active_learning, u):
        assert active_learning.static_values is not None
        return active_learning.static_values[active_learning.candidate_inds]
    
'''
Helper functions from github.com/vwz/AGE -- i.e the source code for AGE
'''
#calculate the percentage of elements smaller than the k-th element
def perc(input,k): return sum([1 if i else 0 for i in input<input[k]])/float(len(input))

#calculate the percentage of elements larger than the k-th element
def percd(input,k): return sum([1 if i else 0 for i in input>input[k]])/float(len(input))

class AGE(acquisition_function):
    '''
    Active learning by Graph Embedding (AGE, Cai 2017).
    '''
    def __init__(self):
        self.iter = 1
        self.basef = 0.9
        
    def compute_values(self, active_learning, u):
        assert active_learning.static_values is not None
        gamma = np.random.beta(1, 1.005-self.basef**self.iter)
        alpha = beta = (1.-gamma)/2.
        
        # AGE code, copied and modified from github.com/vwz/AGE 
        entropy = spentropy(u, axis=1)
        entrperc = np.asarray([perc(entropy,i) for i in range(len(entropy))])
        kmeans = KMeans(n_clusters=u.shape[1], random_state=0).fit(u)
        ed=euclidean_distances(u,kmeans.cluster_centers_)
        ed_score = np.min(ed,axis=1) #the larger ed_score is, the far that node is away from cluster centers, the less representativeness the node is
        edprec = np.asarray([percd(ed_score,i) for i in range(len(ed_score))])
        finalweight = alpha*entrperc + beta*edprec + gamma*active_learning.static_values
        
        # update iteration counter for time-dependent parameters
        self.iter += 1
        
        # return acquisition values only on the candidate indices
        return finalweight[active_learning.candidate_inds]

class voptfull(acquisition_function):
    '''
    FULL Vopt, uses L^{-1}
    '''
        
    def compute_values(self, active_learning, u):
        assert hasattr(active_learning, "fullC")
        vopt_values = np.linalg.norm(active_learning.fullC[:, active_learning.candidate_inds], axis=0)**2.
        vopt_values /= active_learning.fullC[active_learning.candidate_inds, active_learning.candidate_inds].flatten()
        return vopt_values

class uncnormprop_OLD(acquisition_function):
    '''
    Sample proportional to the acquisition function's values.
    
    Currently implemented for SEQUENTIAL only
    '''
    def compute_values(self, active_learning, u):
        vals = 1. - np.linalg.norm(u[active_learning.candidate_inds,:], axis=1)
        
        # scaling for p(x) \propto e^{x/T}, where T is scales as the values change
        T = vals.max() - np.percentile(vals, 90)
        T = max(0.01, min(1,T))
        p = np.exp(vals/T)
        
        # select a point at random according to the probabilities previously calculated and then return standard basis vector of that index
        k_choice = np.random.choice(np.arange(active_learning.candidate_inds.size), p=p/p.sum())
        acq_vals = np.zeros_like(active_learning.candidate_inds)
        acq_vals[k_choice] = 1.
        return acq_vals
    
    
class uncnormprop(acquisition_function):
    '''
    Sample proportional to the acquisition function's values.
    
    Currently implemented for SEQUENTIAL only
    '''
    def __init__(self):
        self.K = 10
        self.log_Eps_tilde = np.log(1e150)  # log of square root of roughly the max precision of python float
    
    def set_K(self, K):
        print(f"Setting K = {K} for uncnormprop")
        self.K = K
        
    def compute_values(self, active_learning, u):
        vals = 1. - np.linalg.norm(u[active_learning.candidate_inds,:], axis=1)
        
        # scaling for p(x) \propto e^{x/T}, where T is scales as the values change. Ensures no numerical overflow occurs
        M = vals.max()
        T0 = M - np.percentile(vals, 100*(1. - 1./self.K))
        eps = M / (self.log_Eps_tilde - np.log(vals.size))
        T = max(eps, min(1.0,T0))
        p = np.exp(vals/T)
        
        # select a point at random according to the probabilities previously calculated and then return standard basis vector of that index
        k_choice = np.random.choice(np.arange(active_learning.candidate_inds.size), p=p/p.sum())
        acq_vals = np.zeros_like(active_learning.candidate_inds)
        acq_vals[k_choice] = 1.
        
        return acq_vals
    
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
        
    def compute_values(self, active_learning, u):
        vals = 1. - np.linalg.norm(u[active_learning.candidate_inds,:], axis=1)
        
        # scaling for p(x) \propto e^{x/T}, where T is scales as the values change. Ensures no numerical overflow occurs
        M = vals.max()
        T0 = M - np.percentile(vals, 100*(1. - 1./self.K))
        eps = M / (self.log_Eps_tilde - np.log(vals.size))
        T = max(eps, min(1.0,T0))
        p = np.exp(vals/T)
        
        # select a batch of points at random according to the probabilities previously calculated and then return standard basis vector of the maximizing index
        k_choices = np.random.choice(np.arange(active_learning.candidate_inds.size), 10, replace=False, p=p/p.sum())
        k_choice = k_choices[np.argmax(vals[k_choices])]
        acq_vals = np.zeros_like(active_learning.candidate_inds)
        acq_vals[k_choice] = 1.
        
        return acq_vals


ACQS = {'unc': uncertainty_sampling(),
        'unckde': uncertainty_sampling(),
        'uncsftmax':uncsftmax(),
        'uncdist':uncdist(),
        'uncsftmaxnorm':uncertainty_sampling("norm"),
        'uncnorm':uncnorm(),
        'uncnormkde':uncnorm(),
        'uncnormdecaytau': uncnorm(),
        'uncnormdecaytaukde': uncnorm(),
        'uncnormswitchK': uncnorm(),
        'uncnormswitchKcheat': uncnorm(),
        'uncnormprop': uncnormprop(),
        'uncnormpropdecaytau': uncnormprop(),
        'uncnormpropswitchK': uncnormprop(),
        'uncnormprop++': uncnormprop_plusplus(),
        'uncnormprop++decaytau': uncnormprop_plusplus(),
        'uncnormprop++switchK': uncnormprop_plusplus(),
        'uncnormprop++kde': uncnormprop_plusplus(),
        'uncnormprop++decaytaukde': uncnormprop_plusplus(),
        'uncnormprop++switchKkde': uncnormprop_plusplus(),
        'vopt':v_opt(),
        'vopt1':v_opt(),
        'sopt':sigma_opt(),
        'voptfull':voptfull(),
        'mc':model_change(),
        'mc1':model_change(),
        'mcvopt':model_change_vopt(),
        'random':random(),
        'pagerank':static_values(),
        'degree':static_values(),
        'age':AGE()
        }
    

