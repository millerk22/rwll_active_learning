import graphlearning as gl
import os
import numpy as np
import scipy.sparse as sparse
from copy import deepcopy


def get_models(G, model_names):
    MODELS = {'poisson':gl.ssl.poisson(G),  # poisson learning
              'laplace':gl.ssl.laplace(G), # laplace learning
              'rwll' : gl.ssl.laplace(G, reweighting='poisson'),
              'rwll1000': gl.ssl.laplace(G, reweighting='poisson', tau=0.1),
              'rwll0100': gl.ssl.laplace(G, reweighting='poisson', tau=0.01),
              'rwll0010': gl.ssl.laplace(G, reweighting='poisson', tau=0.001)
              }

    return [deepcopy(MODELS[name]) for name in model_names]


def load_graph(dataset, metric, numeigs=200, data_dir="data", returnX=False, returnK=False, knn=None):
    X, clusters = gl.datasets.load(dataset.split("-")[0], metric=metric)
    if dataset.split("-")[-1] == 'evenodd':
        labels = clusters % 2
    elif dataset.split("-")[-1][:3] == "mod":
        modnum = int(dataset[-1])
        labels = clusters % modnum
    else:
        labels = clusters

    if dataset.split("-")[0] == 'mstar': # allows for specific train/test set split TODO
        trainset = None
#     elif metric == "hsi":
#         trainset = np.where(labels != 0)[0]
    else:
        trainset = None

    # Construct the similarity graph
    print(f"Constructing similarity graph for {dataset}")
    
    if knn is None:
        knn = 20
        if dataset == 'isolet':
            print("Using knn = 5 for Isolet")
            knn = 5
        elif dataset in ['box', 'blobs']:
            print(f"knn = 100, {dataset}")
            knn = 100
    
    graph_filename = os.path.join(data_dir, f"{dataset.split('-')[0]}_{knn}")

    normalization = "combinatorial"
    method = "lowrank"
    if dataset.split("-")[0] in ["mnist", "fashionmnist", "cifar", "emnist", "mnistsmall", "fashionmnistsmall", "salinassub", "paviasub", "mnistimb", "fashionmnistimb", "emnistvcd"]:
        normalization = "normalized"
    if labels.size < 100000:
        method = "exact"
    
    print(f"Eigendata calculation will be {method}")

    try:
        G = gl.graph.load(graph_filename)
        found = True
    except:
        if metric == "hsi":
            W = gl.weightmatrix.knn(X, knn, similarity="angular") # LAND does 100 in HSI
        else:
            W = gl.weightmatrix.knn(X, knn)
        G = gl.graph(W)
        found = False

    if numeigs is not None:
        eigdata = G.eigendata[normalization]['eigenvalues']
        if eigdata is not None:
            prev_numeigs = eigdata.size
            if prev_numeigs >= numeigs:
                print("Retrieving Eigendata...")
            else:
                print(f"Requested {numeigs} eigenvalues, but have only {prev_numeigs} stored. Recomputing Eigendata...")
        else:
            print(f"No Eigendata found, so computing {numeigs} eigenvectors...")

        evals, evecs = G.eigen_decomp(normalization=normalization, k=numeigs, method=method)


    G.save(graph_filename)
    
    if returnX:
        return G, labels, trainset, normalization, X
    
    if returnK:
        return G, labels, trainset, normalization, np.unique(clusters).size
    
    return G, labels, trainset, normalization


def get_active_learner_eig(G, labeled_ind, labels, acq_func_name, gamma=0.1, normalization='combinatorial'):
    numeigs = 100 # default
    if len(acq_func_name.split("-")) > 1:
        numeigs = int(acq_func_name.split("-")[-1])

    # determine if need to recompute eigenvalues/vectors -- Need to still debug what was going wrong with cached evecs and evals
    recompute = True
    if G.eigendata[normalization]['eigenvalues'] is not None:
        if G.eigendata[normalization]['eigenvalues'].size < numeigs:
            recompute = True
    else:
        recompute = True

    if not recompute:
        # Current gl.active_learning is implemented only to allow for exact eigendata compute for "normalized"
        print(f"Using previously stored {normalization} eigendata with {numeigs} evals for computing {acq_func_name}")
        active_learner = gl.active_learning.active_learning(G, labeled_ind.copy(), labels[labeled_ind], eval_cutoff=None)
        
        active_learner.evals = G.eigendata[normalization]['eigenvalues'][:numeigs]
        active_learner.evecs = G.eigendata[normalization]['eigenvectors'][:,:numeigs] 
        if acq_func_name.split("-")[0][-1] == "1":
            active_learner.evals = active_learner.evals[1:numeigs] 
            active_learner.evecs = active_learner.evecs[:,1:numeigs]
        
        active_learner.gamma = gamma
        active_learner.cov_matrix = np.linalg.inv(np.diag(active_learner.evals) + active_learner.evecs[active_learner.current_labeled_set,:].T @ active_learner.evecs[active_learner.current_labeled_set,:] / active_learner.gamma**2.)
        active_learner.init_cov_matrix = active_learner.cov_matrix
    else:
        print("Warning: Computing eigendata with gl.active_learning defaults...")
        active_learner = gl.active_learning.active_learning(G, labeled_ind.copy(), labels[labeled_ind], \
                    eval_cutoff=numeigs, gamma=gamma)
        
        
        if acq_func_name.split("-")[0][-1] == "1":
            active_learner.evals = active_learner.evals[1:] 
            active_learner.evecs = active_learner.evecs[:,1:]
            active_learner.cov_matrix = np.linalg.inv(np.diag(active_learner.evals) +  active_learner.evecs[active_learner.current_labeled_set,:].T @ active_learner.evecs[active_learner.current_labeled_set,:] / active_learner.gamma**2.)
            active_learner.init_cov_matrix = active_learner.cov_matrix
        
    print(f"{acq_func_name}, {active_learner.evals.size}")
    return active_learner



def get_active_learner(acq_func_name, G, labels, labeled_ind, normalization, args):
    """
        Based on the acquisition function name, determine if need to compute the covariance matrix for instantiating 
        active_learner object
    """
    if acq_func_name.split("-")[0] in ["mc", "mcvopt", "vopt", "vopt1", "sopt"]:
        print(f"gamma = {args.gamma}")
        active_learner = get_active_learner_eig(deepcopy(G), labeled_ind.copy(), labels, acq_func_name, 
                                                gamma=args.gamma, normalization=normalization)

    else:
        active_learner = gl.active_learning.active_learning(deepcopy(G), labeled_ind.copy(), 
                                                            labels[labeled_ind], eval_cutoff=None)
        
        # special case of needing to calculate FULL covariance matrix for voptfull
        if acq_func_name in ["voptfull", "soptfull"]:
            print("Calculating fullC")
            active_learner.fullC = sparse.linalg.inv(sparse.csc_matrix(G.laplacian(normalization=normalization) + 
                                                                       0.001*sparse.eye(G.num_nodes))).toarray()
    
    return active_learner


def get_graph_and_models(acq_funcs_names, model_names, args):
    '''
        NEED TO CHANGE AFTER NORMALIZATION TEST
    '''
    
    # Determine if we need to calculate more eigenvectors/values for mc, vopt, mcvopt acquisitions
    maxnumeigs = 0
    for acq_func_name in acq_funcs_names:
#         if acq_func_name in ["mc", "mcvopt", "vopt", "vopt1"]:
#             maxnumeigs = max(maxnumeigs, 50)
            
        if len(acq_func_name.split("-")) == 1:
            continue
        d = acq_func_name.split("-")[-1]
        if len(d) > 0:
            if maxnumeigs < int(d):
                maxnumeigs = int(d)
    if maxnumeigs == 0:
        maxnumeigs = None

    # Load in the graph and labels
    print("Loading in Graph...")
    G, labels, trainset, normalization, K = load_graph(args.dataset, args.metric, maxnumeigs, returnK=True, knn=args.knn)
    
    models = get_models(G, model_names)
    
    return G, labels, trainset, normalization, models, K
