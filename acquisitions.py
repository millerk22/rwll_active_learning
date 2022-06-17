from graphlearning.active_learning import acquisition_function, model_change, v_opt, model_change_vopt, uncertainty_sampling
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


class betavar(acquisition_function):
    '''
    Beta Learning Variance

    Note: u now is actually the matrix A in beta learning. Need to ensure
    '''
    def compute_values(self, active_learning, u):
        a0 = u.sum(axis=1)
        a = (u * u).sum(axis=1)
        return ((1. - a/(a0**2.))/(1. + a0))[active_learning.candidate_inds]

    
class betavarprop(acquisition_function):
    '''
    Beta Learning Variance, with proportional sampling, not max.
    
    CURRENTLY ONLY IMPLEMENTED FOR SEQUENTIAL, NOT BATCH
    '''
    def __init__(self, percentile=80):
        self.percentile = percentile
        
    def compute_values(self, active_learning, u):
        a0 = u.sum(axis=1)
        a = (u * u).sum(axis=1)
        acq_vals = ((1. - a/(a0**2.))/(1. + a0))[active_learning.candidate_inds]
        
        # do proportional sampling herein to choose a point k_choice that is not necessarily the maximizer
        inds = np.where(acq_vals >= np.percentile(acq_vals, self.percentile))[0]
        k_choice = np.random.choice(inds) # uniform over the top (100 - self.percentile)% of points
        
        # return values so that this k_choice will be the maximizer
        return_vals = np.zeros_like(acq_vals)
        return_vals[k_choice] = 1.0
        
        return return_vals

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

ACQS = {'unc': uncertainty_sampling(),
        'uncsftmax':uncsftmax(),
        'uncdist':uncdist(),
        'uncsftmaxnorm':uncertainty_sampling("norm"),
        'uncnorm':uncnorm(),
        'vopt':v_opt(),
        'mc':model_change(),
        'mcvopt':model_change_vopt(),
        'random':random(),
        'betavar':betavar(),
        'betavarprop':betavarprop(),
        'betavarprop70':betavarprop(70),
        'pagerank':static_values(),
        'degree':static_values(),
        'age':AGE()
        }
    

