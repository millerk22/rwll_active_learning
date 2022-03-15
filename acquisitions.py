from graphlearning.active_learning import acquisition_function, model_change, v_opt, model_change_vopt, uncertainty_sampling
import numpy as np
from scipy.special import softmax




class uncsftmax(acquisition_function):
    '''
    Smallest margin on softmax of the output, u
    '''
    def compute_values(active_learning, u):
        s = softmax(u, axis=1)
        u_sort = np.sort(s)
        return 1. - (u_sort[active_learning.candidate_inds,-1] - u_sort[active_learning.candidate_inds,-2]) # smallest margin

class uncdist(acquisition_function):
    '''
    Straightforward Euclidean distance to current pseudolabel
    '''
    def compute_values(active_learning, u):
        one_hot_predicted_labels = np.eye(u.shape[1])[np.argmax(u, axis=1)]
        return  np.linalg.norm((u - one_hot_predicted_labels), axis=1)[active_learning.candidate_inds]


class uncnorm(acquisition_function):
    '''
    Norm of the output rows in u.
    '''
    def compute_values(active_learning, u):
        return 1. - np.linalg.norm(u[active_learning.candidate_inds], axis=1)


class betavar(acquisition_function):
    '''
    Beta Learning Variance

    Note: u now is actually the matrix A in beta learning. Need to ensure
    '''
    def compute_values(active_learning, u):
        a0 = u.sum(axis=1)
        a = (u * u).sum(axis=1)
        return ((1. - a/(a0**2.))/(1. + a0))[active_learning.candidate_inds]

class random(acquisition_function):
    '''
    Random choices
    '''
    def compute_values(active_learning, u):
        return np.random.rand(u.shape[0])[active_learning.candidate_inds]


ACQS = {'unc': uncertainty_sampling(),
        'uncsftmax':uncsftmax,
        'uncdist':uncdist,
        'uncsftmaxnorm':uncertainty_sampling("norm"),
        'uncnorm':uncnorm,
        'vopt':v_opt(),
        'mc':model_change(),
        'mcvopt':model_change_vopt(),
        'random':random,
        'betavar':betavar}
