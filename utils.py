import graphlearning as gl
import os
import numpy as np

def load_graph(dataset, metric, numeigs=200, data_dir="data", returnX=False):
    X, labels = gl.datasets.load(dataset.split("-")[0], metric=metric)
    if dataset.split("-")[-1] == 'evenodd':
        labels = labels % 2
    elif dataset.split("-")[-1][:3] == "mod":
        modnum = int(dataset[-1])
        labels = labels % modnum

    if dataset.split("-")[0] == 'mstar': # allows for specific train/test set split TODO
        trainset = None
#     elif metric == "hsi":
#         trainset = np.where(labels != 0)[0]
    else:
        trainset = None

    # Construct the similarity graph
    print(f"Constructing similarity graph for {dataset}")
    knn = 20
    graph_filename = os.path.join(data_dir, f"{dataset.split('-')[0]}_{knn}")

    normalization = "combinatorial"
    method = "lowrank"
    if dataset.split("-")[0] in ["fashionmnist", "cifar", "emnist"]:
        normalization = "normalized"
    if labels.size < 10000:
        method = "exact"

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

        evals, evecs = G.eigen_decomp(normalization=normalization, k=numeigs, method=method, q=150)


    if not found:
        G.save(graph_filename)
    
    if returnX:
        return G, labels, trainset, normalization, X
    return G, labels, trainset, normalization


def get_active_learner_eig(G, labeled_ind, labels, acq_func_name, gamma=0.1, normalization='combinatorial'):
    numeigs = 50 # default
    if len(acq_func_name.split("-")) > 1:
        numeigs = int(acq_func_name.split("-")[-1])

    # determine if need to recompute eigenvalues/vectors
    recompute = False
    if G.eigendata[normalization]['eigenvalues'] is not None:
        if G.eigendata[normalization]['eigenvalues'].size < numeigs:
            recompute = True
    else:
        recompute = True

    if not recompute:
        # Current gl.active_learning is implemented only to allow for exact eigendata compute for "normalized"
        print(f"Using previously stored {normalization} eigendata with {numeigs} evals for computing {acq_func_name}")
        active_learner = gl.active_learning.active_learning(G, labeled_ind.copy(), labels[labeled_ind], eval_cutoff=None)
        active_learner.evals = G.eigendata[normalization]['eigenvalues'] 
        active_learner.evecs = G.eigendata[normalization]['eigenvectors'] 
        active_learner.gamma = gamma
        active_learner.cov_matrix = np.linalg.inv(np.diag(active_learner.evals) + active_learner.evecs[active_learner.current_labeled_set,:].T @ active_learner.evecs[active_learner.current_labeled_set,:] / active_learner.gamma**2.)
        active_learner.init_cov_matrix = active_learner.cov_matrix
    else:
        print("Warning: Computing eigendata with gl.active_learning defaults...")
        active_learner = gl.active_learning.active_learning(G, labeled_ind.copy(), labels[labeled_ind], \
                    eval_cutoff=numeigs, gamma=gamma)

    return active_learner

