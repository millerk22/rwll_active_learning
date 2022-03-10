import numpy as np
from scipy.special import softmax


def beta_var(A):
    a0 = A.sum(axis=1)
    a = (A * A).sum(axis=1)
    return ((1. - a/(a0**2.))/(1. + a0))

def unc(u):
    u_sort = np.sort(u)
    return 1. - (u_sort[:,-1] - u_sort[:,-2]) # smallest margin acquisition function

def uncsftmax(u):
    s = softmax(u, axis=1)
    u_sort = np.sort(s)
    return 1. - (u_sort[:,-1] - u_sort[:,-2]) # smallest margin

def uncdist(u):
    '''
    Straightforward Euclidean distance to current pseudolabel
    '''
    one_hot_predicted_labels = np.eye(u.shape[1])[np.argmax(u, axis=1)]
    return  np.linalg.norm((u - one_hot_predicted_labels), axis=1)

def uncsftmaxnorm(u):
    '''
    Project onto simplex and then Euclidean distance to current pseudolabel
    '''
    u_probs = softmax(u, axis=1)
    one_hot_predicted_labels = np.eye(u.shape[1])[np.argmax(u, axis=1)]
    return np.linalg.norm((u_probs - one_hot_predicted_labels), axis=1)

def uncnorm(u):
    return 1. - np.linalg.norm(u, axis=1)

def vopt(u, C_a, evecs, gamma=0.1):
    Cavk = C_a @ evecs.T
    col_norms = np.linalg.norm(Cavk, axis=0)
    diag_terms = (gamma**2. + np.array([np.inner(evecs[k,:], Cavk[:, i]) for i,k in enumerate(np.arange(u.shape[0]))]))
    return col_norms**2. / diag_terms

def mc(u, C_a, evecs, gamma=0.1):
    Cavk = C_a @ evecs.T
    col_norms = np.linalg.norm(Cavk, axis=0)
    diag_terms = (gamma**2. + np.array([np.inner(evecs[k,:], Cavk[:, i]) for i,k in enumerate(np.arange(u.shape[0]))]))
    unc_terms = uncdist(u) # straightforward distance to current pseudolabel
    return unc_terms * col_norms / diag_terms

def mcvopt(u, C_a, evecs, gamma=0.1):
    Cavk = C_a @ evecs.T
    col_norms = np.linalg.norm(Cavk, axis=0)
    diag_terms = (gamma**2. + np.array([np.inner(evecs[k,:], Cavk[:, i]) for i,k in enumerate(np.arange(u.shape[0]))]))
    unc_terms = uncdist(u) # straightforward distance to current pseudolabel
    return unc_terms * col_norms **2. / diag_terms

def random(u):
    return np.random.rand(u.shape[0])
